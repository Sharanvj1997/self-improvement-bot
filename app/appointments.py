"""Deterministic appointment service. All business rules live here, never in the prompt.

Every function returns {"ok": True, ...} or {"ok": False, "error": CODE, "message": str}.
"""
import re
import sqlite3
from datetime import date, timedelta

from app.config import DOCTOR, HORIZON_DAYS, SLOT_TIMES, TODAY


def _ok(**kw) -> dict:
    return {"ok": True, **kw}


def _err(code: str, message: str, **kw) -> dict:
    return {"ok": False, "error": code, "message": message, **kw}


def normalize_phone(phone: str) -> str | None:
    """Return a 10-digit number, or None if invalid. Accepts +91 / spaces / dashes."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return digits if len(digits) == 10 else None


def _check_slot(d: str, t: str) -> dict | None:
    """Return an error dict if (date, time) is not a bookable slot, else None."""
    try:
        day = date.fromisoformat(d)
    except ValueError:
        return _err("INVALID_DATE", f"'{d}' is not a valid YYYY-MM-DD date.")
    if day < TODAY:
        return _err("PAST_DATE", "That date is in the past.")
    if day > TODAY + timedelta(days=HORIZON_DAYS):
        return _err("BEYOND_HORIZON", f"We only book up to {HORIZON_DAYS} days ahead.")
    if day.weekday() >= 5:
        return _err("NOT_A_WORKDAY", "The clinic is closed on weekends.")
    if t not in SLOT_TIMES:
        return _err("INVALID_TIME", f"'{t}' is not a bookable time.", valid_times=SLOT_TIMES)
    return None


def _free_times(conn: sqlite3.Connection, d: str) -> list[str]:
    rows = conn.execute(
        "SELECT time FROM appointments WHERE doctor=? AND date=? AND status='booked'",
        (DOCTOR, d),
    ).fetchall()
    taken = {r["time"] for r in rows}
    return [t for t in SLOT_TIMES if t not in taken]


def _next_open_slots(conn: sqlite3.Connection, start: str, limit: int = 3) -> list[dict]:
    """Nearest open slots on/after `start`. Used to offer alternatives."""
    out: list[dict] = []
    day = date.fromisoformat(start)
    for _ in range(HORIZON_DAYS + 1):
        if TODAY <= day <= TODAY + timedelta(days=HORIZON_DAYS) and day.weekday() < 5:
            for t in _free_times(conn, day.isoformat()):
                out.append({"date": day.isoformat(), "time": t})
                if len(out) == limit:
                    return out
        day += timedelta(days=1)
    return out


def check_availability(conn: sqlite3.Connection, d: str, t: str | None = None) -> dict:
    err = _check_slot(d, t or SLOT_TIMES[0])
    if err:
        return err
    free = _free_times(conn, d)
    if t is None:
        return _ok(date=d, available_times=free,
                   alternatives=[] if free else _next_open_slots(conn, d))
    if t in free:
        return _ok(date=d, time=t, available=True)
    return _ok(date=d, time=t, available=False, available_times=free,
               alternatives=_next_open_slots(conn, d))


def book_appointment(conn: sqlite3.Connection, patient_name: str, phone: str,
                     d: str, t: str) -> dict:
    if not patient_name.strip():
        return _err("MISSING_NAME", "Patient name is required.")
    p = normalize_phone(phone)
    if p is None:
        return _err("INVALID_PHONE", "Phone must be a 10-digit number.")
    err = _check_slot(d, t)
    if err:
        return err
    try:
        cur = conn.execute(
            "INSERT INTO appointments(patient_name, patient_phone, doctor, date, time, status)"
            " VALUES (?,?,?,?,?, 'booked')",
            (patient_name.strip(), p, DOCTOR, d, t),
        )
        conn.commit()
    except sqlite3.IntegrityError:  # the unique index fired: slot already taken
        conn.rollback()
        return _err("SLOT_UNAVAILABLE", f"{d} {t} is already booked.",
                    alternatives=_next_open_slots(conn, d))
    return _ok(appointment_id=cur.lastrowid, doctor=DOCTOR, date=d, time=t)


def _owned_active(conn: sqlite3.Connection, appt_id: int,
                  phone: str) -> tuple[sqlite3.Row | None, dict | None]:
    """Fetch an active appointment the phone owns, or return an error dict."""
    p = normalize_phone(phone)
    if p is None:
        return None, _err("INVALID_PHONE", "Phone must be a 10-digit number.")
    row = conn.execute("SELECT * FROM appointments WHERE id=?", (appt_id,)).fetchone()
    if row is None or row["status"] != "booked":
        return None, _err("NOT_FOUND", "No active appointment with that id.")
    if row["patient_phone"] != p:
        return None, _err("NOT_OWNER", "That appointment does not belong to this patient.")
    return row, None


def cancel_appointment(conn: sqlite3.Connection, appt_id: int, phone: str) -> dict:
    row, err = _owned_active(conn, appt_id, phone)
    if err:
        return err
    conn.execute("UPDATE appointments SET status='cancelled' WHERE id=?", (appt_id,))
    conn.commit()
    return _ok(appointment_id=appt_id, cancelled=True, date=row["date"], time=row["time"])


def reschedule_appointment(conn: sqlite3.Connection, appt_id: int, phone: str,
                           new_d: str, new_t: str) -> dict:
    """Atomic: a single UPDATE. If the new slot is taken, the unique index rejects it
    and the original appointment is untouched."""
    row, err = _owned_active(conn, appt_id, phone)
    if err:
        return err
    err = _check_slot(new_d, new_t)
    if err:
        return err
    if (row["date"], row["time"]) == (new_d, new_t):
        return _err("SAME_SLOT", "That is already the appointment's slot.")
    try:
        conn.execute("UPDATE appointments SET date=?, time=? WHERE id=?",
                     (new_d, new_t, appt_id))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        return _err("SLOT_UNAVAILABLE",
                    f"{new_d} {new_t} is already booked; original appointment kept.",
                    alternatives=_next_open_slots(conn, new_d))
    return _ok(appointment_id=appt_id,
               old={"date": row["date"], "time": row["time"]},
               new={"date": new_d, "time": new_t})


def get_patient_appointments(conn: sqlite3.Connection, phone: str) -> dict:
    p = normalize_phone(phone)
    if p is None:
        return _err("INVALID_PHONE", "Phone must be a 10-digit number.")
    rows = conn.execute(
        "SELECT id, doctor, date, time FROM appointments"
        " WHERE patient_phone=? AND status='booked' ORDER BY date, time", (p,)
    ).fetchall()
    return _ok(appointments=[dict(r) for r in rows])