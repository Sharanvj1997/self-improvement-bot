import sqlite3
import pytest
from app.database import fresh_seeded_db


def test_seed_data_loaded():
    conn = fresh_seeded_db()
    assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 2


def test_unique_index_blocks_double_booking():
    conn = fresh_seeded_db()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO appointments(patient_name,patient_phone,doctor,date,time,status)"
            " VALUES ('X','9333333333','Dr. Meera Rao','2026-10-06','10:00','booked')"
        )


def test_cancelled_slot_can_be_rebooked():
    conn = fresh_seeded_db()
    conn.execute("UPDATE appointments SET status='cancelled' WHERE id=1")
    conn.execute(
        "INSERT INTO appointments(patient_name,patient_phone,doctor,date,time,status)"
        " VALUES ('X','9333333333','Dr. Meera Rao','2026-10-06','10:00','booked')"
    )