from app.guardrails import load_guardrails
from app.llm import LLMResponse
from evaluation.evaluator import ScenarioResult, SuiteResult
from evaluation.regression import accept, compare, improvement_loop, run_suite
from evaluation.schema import Scenario
from tests.test_agent import ScriptedLLM, call, text
from tests.test_improvement import GOOD
import pytest


def suite(*items):
    return SuiteResult([ScenarioResult(n, [], float(s), p) for n, p, s in items])


def test_compare_classifies_every_scenario():
    before = suite(("a", True, 100), ("b", False, 50), ("c", True, 100), ("d", False, 40))
    after = suite(("a", True, 100), ("b", True, 100), ("c", False, 60), ("d", False, 40))
    c = compare(before, after)
    assert c.fixed == ["b"] and c.regressions == ["c"]
    assert c.still_passing == ["a"] and c.still_failing == ["d"]
    assert c.delta == 2.5 and not accept(c)  # score rose, but a regression -> rejected


def test_accept_requires_no_regression_and_no_score_drop():
    same = suite(("a", True, 100))
    assert accept(compare(same, same))
    assert not accept(compare(same, suite(("a", True, 90))))


def test_repeats_fail_a_scenario_if_any_pass_fails_and_average_the_score():
    sc = Scenario(name="s", description="d", patient_turns=["hi"],
                  checks=[{"type": "tool_not_called", "name": "book_appointment"}])
    book = call("t", "book_appointment", patient_name="Asha", phone="9876543210",
                date="2026-10-07", time="11:00")
    llm = ScriptedLLM([text("Hello"), book, text("Done.")])  # pass 1 fine, pass 2 books
    agg, _ = run_suite([sc], llm, [], repeats=2, judge=None)
    r = agg.scenarios[0]
    assert not r.passed and r.score == 50.0
    assert any(c.passed is False for c in r.checks)  # representative run is the failing one


class FakeLLM:
    """Naive agent acts immediately. With LEARNED RULES it asks first.
    breaks_thursday: with rules it stops handling Thursday requests (simulates a regression)."""

    def __init__(self, breaks_thursday=False):
        self.breaks_thursday = breaks_thursday

    def complete(self, system, messages, tools):
        if system.startswith("You write precise"):  # the guardrail-drafting call
            return LLMResponse(content=text(GOOD))
        last = messages[-1]
        if last["role"] == "user" and not isinstance(last["content"], str):
            return LLMResponse(content=text("Done."))  # a tool result just came back
        users = [m for m in messages if m["role"] == "user" and isinstance(m["content"], str)]
        thursday = "thu" in users[0]["content"]
        careful = "LEARNED RULES" in system
        if careful and self.breaks_thursday and thursday:
            return LLMResponse(content=text("Hmm, not sure."))
        if careful and len(users) < 2:
            return LLMResponse(content=text("Shall I book it?"))
        phone, d, t = (("9111111111", "2026-10-08", "15:00") if thursday
                       else ("9876543210", "2026-10-07", "11:00"))
        return LLMResponse(content=call("c", "book_appointment", patient_name="Asha",
                                        phone=phone, date=d, time=t))


S1 = Scenario(name="s1", description="Must confirm before booking.",
              patient_turns=["book wed 11", "yes"],
              checks=[{"type": "no_mutation_before_turn", "turn": 2},
                      {"type": "db_booked", "phone": "9876543210", "date": "2026-10-07", "time": "11:00"}])
S2 = Scenario(name="s2", description="Thursday booking.",
              patient_turns=["book thu 3pm", "yes"],
              checks=[{"type": "db_booked", "phone": "9111111111", "date": "2026-10-08", "time": "15:00"}])


def test_loop_fixes_failure_without_regression_and_keeps_guardrails(tmp_path):
    path = tmp_path / "g.json"
    result = improvement_loop([S1, S2], FakeLLM(), path, results_dir=tmp_path / "res")
    assert result.comparison.fixed == ["s1"] and result.comparison.regressions == []
    assert result.accepted and result.after.score > result.before.score
    learned = load_guardrails(path)
    assert [g.failure_type for g in learned] == ["action_without_confirmation"]
    assert learned[0].source_scenario == "s1"  # traceable to the failed scenario
    assert (tmp_path / "res" / "before.json").exists() and (tmp_path / "res" / "after.json").exists()


def test_loop_rolls_back_when_an_improvement_causes_a_regression(tmp_path):
    path = tmp_path / "g.json"
    path.write_text("[]")
    result = improvement_loop([S1, S2], FakeLLM(breaks_thursday=True), path)
    assert result.comparison.regressions == ["s2"]
    assert not result.accepted
    assert load_guardrails(path) == []  # rolled back to the original file



from evaluation.regression import HOLDOUT_PATH
from evaluation.schema import load_scenarios


def test_holdout_is_valid_and_disjoint_from_training():
    train, hold = load_scenarios(), load_scenarios(HOLDOUT_PATH)
    assert len(hold) >= 5
    assert {s.name for s in train}.isdisjoint({s.name for s in hold})
    assert {s.patient_turns[0] for s in train}.isdisjoint({s.patient_turns[0] for s in hold})


def test_holdout_covers_the_learned_failure_types():
    types = {c.resolved_failure_type() for s in load_scenarios(HOLDOUT_PATH) for c in s.checks}
    assert {"ambiguous_date", "action_without_confirmation", "missing_urgent_care_referral",
            "weekday_date_mismatch"} <= types




class DownLLM:
    def complete(self, *a, **k):
        raise RuntimeError("quota exhausted")


def test_loop_aborts_and_restores_guardrails_when_provider_is_down(tmp_path):
    path = tmp_path / "g.json"
    path.write_text('[{"failure_type": "x", "lesson": "l", "guardrail": "g", '
                    '"priority": "high", "source_scenario": "s"}]')
    original = path.read_text()
    with pytest.raises(SystemExit):
        improvement_loop([S1], DownLLM(), path)
    assert path.read_text() == original  # nothing wiped, nothing learned from errors


def test_daily_quota_error_is_not_retried():
    from app.llm import OpenAICompatClient

    class Boom(Exception):
        status_code = 429

    calls = []

    def fake_create(**kwargs):
        calls.append(1)
        raise Boom("Rate limit reached on tokens per day (TPD)")

    c = OpenAICompatClient("m", "k", "http://localhost:1")
    c._client.chat.completions.create = fake_create
    with pytest.raises(Boom):
        c.complete("s", [{"role": "user", "content": "hi"}], [])
    assert len(calls) == 1