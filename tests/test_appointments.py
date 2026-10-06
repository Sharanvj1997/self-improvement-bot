import pytest
from app import appointments as svc
from app.database import fresh_seeded_db


@pytest.fixture
def conn():
    return fresh_seeded_db()


def test_availability_lists_free_times(conn):
    r = svc.check_availability(conn, "2026-10-07")
    assert r["ok"] and "10:00" in r["available_times"]


def test_seeded_slot_unavailable_with_alternatives(conn):
    r = svc.check_availability(conn, "2026-10-06", "10:00")
    assert r["ok"] and r["available"] is False and r["alternatives"]
    assert {"date": "2026-10-06", "time": "10:00"} not in r["alternatives"]


def test_successful_booking(conn):
    r = svc.book_appointment(conn, "Asha K", "9876543210", "2026-10-07", "10:00")
    assert r["ok"]
    assert svc.check_availability(conn, "2026-10-07", "10:00")["available"] is False


def test_booking_unavailable_slot_rejected(conn):
    r = svc.book_appointment(conn, "Asha K", "9876543210", "2026-10-06", "10:00")
    assert not r["ok"] and r["error"] == "SLOT_UNAVAILABLE"


def test_double_booking_prevented(conn):
    assert svc.book_appointment(conn, "A", "9111111111", "2026-10-07", "09:00")["ok"]
    r = svc.book_appointment(conn, "B", "9222222222", "2026-10-07", "09:00")
    assert r["error"] == "SLOT_UNAVAILABLE"


@pytest.mark.parametrize("d,t,code", [
    ("2026-10-04", "10:00", "PAST_DATE"),
    ("2026-10-10", "10:00", "NOT_A_WORKDAY"),   # Saturday
    ("2026-10-07", "13:00", "INVALID_TIME"),    # lunch
    ("next friday", "10:00", "INVALID_DATE"),   # LLM must resolve dates itself
    ("2027-01-01", "10:00", "BEYOND_HORIZON"),
])
def test_slot_validation(conn, d, t, code):
    assert svc.book_appointment(conn, "A", "9111111111", d, t)["error"] == code


def test_invalid_phone(conn):
    assert svc.book_appointment(conn, "A", "123", "2026-10-07", "10:00")["error"] == "INVALID_PHONE"



def test_cancel_frees_slot(conn):
    r = svc.cancel_appointment(conn, 2, "9000000002")
    assert r["ok"]
    assert svc.check_availability(conn, "2026-10-08", "11:00")["available"] is True


def test_cancel_wrong_owner_rejected(conn):
    assert svc.cancel_appointment(conn, 2, "9999999999")["error"] == "NOT_OWNER"
    assert svc.get_patient_appointments(conn, "9000000002")["appointments"]  # untouched


def test_cancel_unknown_id(conn):
    assert svc.cancel_appointment(conn, 999, "9000000002")["error"] == "NOT_FOUND"


def test_reschedule_success(conn):
    r = svc.reschedule_appointment(conn, 2, "9000000002", "2026-10-09", "15:00")
    assert r["ok"]
    assert svc.check_availability(conn, "2026-10-08", "11:00")["available"] is True


def test_reschedule_is_atomic_when_target_taken(conn):
    r = svc.reschedule_appointment(conn, 2, "9000000002", "2026-10-06", "10:00")
    assert r["error"] == "SLOT_UNAVAILABLE"
    appts = svc.get_patient_appointments(conn, "9000000002")["appointments"]
    assert appts[0]["date"] == "2026-10-08" and appts[0]["time"] == "11:00"


def test_get_patient_appointments(conn):
    r = svc.get_patient_appointments(conn, "9000000002")
    assert r["ok"] and len(r["appointments"]) == 1