from app import appointments as svc
from app.agent import FALLBACK, MAX_TOOL_ROUNDS, Agent
from app.database import fresh_seeded_db
from app.llm import LLMResponse


class ScriptedLLM:
    """Returns pre-written responses in order (repeats the last one if exhausted)."""
    def __init__(self, script):
        self.script, self.i = script, 0

    def complete(self, system, messages, tools):
        r = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return LLMResponse(content=r)


def text(t):
    return [{"type": "text", "text": t}]


def call(id_, name, **args):
    return [{"type": "tool_use", "id": id_, "name": name, "input": args}]


def test_tool_call_executes_in_db_and_is_logged():
    conn = fresh_seeded_db()
    llm = ScriptedLLM([
        call("t1", "book_appointment", patient_name="Asha", phone="9876543210",
             date="2026-10-07", time="10:00"),
        text("Booked!"),
    ])
    agent = Agent(llm, conn)
    assert agent.respond("book me wed 10am") == "Booked!"
    assert svc.check_availability(conn, "2026-10-07", "10:00")["available"] is False
    ev = agent.state.tool_log[0]
    assert ev.name == "book_appointment" and ev.result["ok"] and ev.patient_turn == 1


def test_failed_tool_result_is_logged_and_nothing_booked():
    conn = fresh_seeded_db()
    llm = ScriptedLLM([
        call("t1", "book_appointment", patient_name="Asha", phone="9876543210",
             date="2026-10-06", time="10:00"),
        text("Sorry, that slot is taken."),
    ])
    agent = Agent(llm, conn)
    agent.respond("book tue 10")
    assert agent.state.tool_log[0].result["error"] == "SLOT_UNAVAILABLE"


def test_tool_loop_is_bounded():
    llm = ScriptedLLM([call("t", "check_availability", date="2026-10-07")])  # never stops
    agent = Agent(llm, fresh_seeded_db())
    assert agent.respond("hi") == FALLBACK
    assert len(agent.state.tool_log) == MAX_TOOL_ROUNDS


def test_patient_turn_index_increments():
    llm = ScriptedLLM([text("a"), call("t", "check_availability", date="2026-10-07"), text("b")])
    agent = Agent(llm, fresh_seeded_db())
    agent.respond("one")
    agent.respond("two")
    assert agent.state.tool_log[0].patient_turn == 2


from app.agent import add_weekdays


def test_add_weekdays_annotates_nested_dates():
    r = add_weekdays({"ok": True, "date": "2026-10-16",
                      "alternatives": [{"date": "2026-10-16", "time": "09:00"}],
                      "old": {"date": "2026-10-08"}})
    assert r["date_weekday"] == "Friday"
    assert r["alternatives"][0]["date_weekday"] == "Friday"
    assert r["old"]["date_weekday"] == "Thursday"


def test_add_weekdays_ignores_bad_values():
    data = {"date": "not-a-date", "n": 3, "date2": "2026-13-45"}
    assert add_weekdays(data) == data