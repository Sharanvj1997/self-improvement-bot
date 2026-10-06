from app.llm import LLMResponse
from evaluation.runner import run_scenario
from evaluation.schema import DbBooked, Scenario
from tests.test_agent import ScriptedLLM, call, text


def make_scenario() -> Scenario:
    return Scenario(
        name="t", description="d",
        patient_turns=["hi", "I'm Asha, 9876543210", "yes"],
        checks=[DbBooked(type="db_booked", phone="9876543210", date="2026-10-07", time="11:00")],
    )


def make_llm() -> ScriptedLLM:
    return ScriptedLLM([
        text("Name and phone?"),
        call("t1", "book_appointment", patient_name="Asha", phone="9876543210",
             date="2026-10-07", time="11:00"),
        text("Booked."),
        text("Anything else?"),
    ])


def booked_count(run) -> int:
    return run.conn.execute(
        "SELECT COUNT(*) FROM appointments WHERE patient_phone='9876543210' AND status='booked'"
    ).fetchone()[0]


def test_runner_plays_all_turns_and_returns_final_db():
    run = run_scenario(make_scenario(), make_llm())
    assert run.error is None
    assert len(run.state.transcript) == 6  # 3 patient + 3 agent
    assert booked_count(run) == 1
    assert run.state.tool_log[0].patient_turn == 2


def test_each_run_gets_a_fresh_db():
    a = run_scenario(make_scenario(), make_llm())
    b = run_scenario(make_scenario(), make_llm())
    total = lambda r: r.conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
    assert total(a) == 3 and total(b) == 3  # 2 seed rows + 1 booking, no leakage


def test_llm_failure_is_recorded_not_raised():
    class BoomLLM:
        def complete(self, system, messages, tools) -> LLMResponse:
            raise RuntimeError("boom")

    run = run_scenario(make_scenario(), BoomLLM())
    assert run.error is not None and "boom" in run.error