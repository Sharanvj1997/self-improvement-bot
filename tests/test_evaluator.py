import pytest

from evaluation.evaluator import evaluate_run
from evaluation.runner import run_scenario
from evaluation.schema import Scenario
from tests.test_agent import ScriptedLLM, call, text


def BOOK():
    return call("t1", "book_appointment", patient_name="Asha", phone="9876543210",
                date="2026-10-07", time="11:00")


BOOKED = {"type": "db_booked", "phone": "9876543210", "date": "2026-10-07", "time": "11:00"}
NO_MUT = {"type": "no_mutation_before_turn", "turn": 2}


def run_for(turns, checks, script):
    sc = Scenario(name="t", description="d", patient_turns=turns, checks=checks)
    return run_scenario(sc, ScriptedLLM(script))


def ev(turns, checks, script):
    return evaluate_run(run_for(turns, checks, script))


def test_clean_booking_passes_with_full_score():
    r = ev(["book wed 11"], [BOOKED, {"type": "no_false_success"}],
           [BOOK(), text("Your appointment is confirmed for Wednesday.")])
    assert r.passed and r.score == 100.0


def test_booking_before_confirmation_is_caught():
    assert not ev(["book", "yes"], [NO_MUT], [BOOK(), text("Done."), text("ok")]).passed
    assert ev(["book", "yes"], [NO_MUT], [text("Shall I book it?"), BOOK(), text("Booked.")]).passed


def test_hallucinated_success_is_caught():
    chk = [{"type": "no_false_success"}]
    assert not ev(["book"], chk, [text("Your appointment is booked!")]).passed
    assert ev(["book"], chk, [text("Sorry, that slot is already booked. Another time?")]).passed


def test_weekday_date_consistency():
    chk = [{"type": "weekday_date_consistent"}]
    assert not ev(["x"], chk, [text("See you Monday, 2026-10-16 at 10:00.")]).passed
    assert ev(["x"], chk, [text("See you Friday, 2026\u201110\u201116 at 10:00.")]).passed
    assert ev(["x"], chk, [text("See you Tuesday, Oct 6 at 9am.")]).passed
    assert not ev(["x"], chk, [text("See you Thursday, Oct 6 at 9am.")]).passed


def test_weighted_score_uses_category_weights():
    # outcome passes (30), confirmation fails (0): 30 / (30 + 15) = 66.7%
    r = ev(["book", "yes"], [BOOKED, NO_MUT], [BOOK(), text("Done."), text("ok")])
    assert not r.passed
    assert r.score == pytest.approx(66.7, abs=0.1)


def test_run_error_counts_as_failed_scenario():
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("boom")

    sc = Scenario(name="t", description="d", patient_turns=["hi"], checks=[NO_MUT])
    r = evaluate_run(run_scenario(sc, Boom()))
    assert not r.passed and r.score == 0.0 and r.error


def test_rubric_is_skipped_without_judge_and_never_decides_pass_fail():
    chk = [{"type": "tool_not_called", "name": "book_appointment"},
           {"type": "rubric", "criterion": "polite"}]
    run = run_for(["hi"], chk, [text("Hello!")])
    r = evaluate_run(run)
    assert r.passed and [c.passed for c in r.checks] == [True, None]
    r2 = evaluate_run(run, judge=lambda transcript, criterion: (False, "meh"))
    assert r2.passed  # a failing rubric lowers the score but doesn't fail the scenario
    assert r2.score < 100.0