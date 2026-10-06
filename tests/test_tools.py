import pytest
from app.database import fresh_seeded_db
from app.tools import dispatch, tool_specs


@pytest.fixture
def conn():
    return fresh_seeded_db()


def test_specs_cover_all_five_tools():
    assert {s["name"] for s in tool_specs()} == {
        "check_availability", "book_appointment", "cancel_appointment",
        "reschedule_appointment", "get_patient_appointments"}


def test_unknown_tool(conn):
    assert dispatch(conn, "drop_table", {})["error"] == "UNKNOWN_TOOL"


def test_missing_args_returns_structured_error(conn):
    r = dispatch(conn, "book_appointment", {"patient_name": "A", "date": "2026-10-07"})
    assert r["error"] == "INVALID_ARGUMENTS" and set(r["fields"]) == {"phone", "time"}


def test_extra_args_rejected(conn):
    r = dispatch(conn, "check_availability", {"date": "2026-10-07", "sql": "DROP TABLE x"})
    assert r["error"] == "INVALID_ARGUMENTS"


def test_dispatch_happy_path(conn):
    r = dispatch(conn, "book_appointment",
                 {"patient_name": "Asha", "phone": "+91 98765 43210",
                  "date": "2026-10-07", "time": "10:00"})
    assert r["ok"]
    