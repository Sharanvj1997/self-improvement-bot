"""SQLite setup. The DB, not the LLM, is the source of truth for appointment state."""
import sqlite3
from pathlib import Path

from app.config import DOCTOR

SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name  TEXT NOT NULL,
    patient_phone TEXT NOT NULL,
    doctor        TEXT NOT NULL,
    date          TEXT NOT NULL,   -- YYYY-MM-DD
    time          TEXT NOT NULL,   -- HH:MM
    status        TEXT NOT NULL CHECK (status IN ('booked','cancelled'))
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_slot
    ON appointments(doctor, date, time) WHERE status = 'booked';
"""

# Fixed fixtures so scenarios are reproducible.
SEED = [
    ("Rahul Verma", "9000000001", "2026-10-06", "10:00"),  # makes Tue 10:00 unavailable
    ("Priya Nair", "9000000002", "2026-10-08", "11:00"),   # existing appt for cancel/reschedule
]


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def fresh_seeded_db(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Clean, identically seeded DB. Used for every scenario run."""
    if str(path) != ":memory:":
        Path(path).unlink(missing_ok=True)
    conn = connect(path)
    for name, phone, d, t in SEED:
        conn.execute(
            "INSERT INTO appointments(patient_name, patient_phone, doctor, date, time, status)"
            " VALUES (?,?,?,?,?, 'booked')",
            (name, phone, DOCTOR, d, t),
        )
    conn.commit()
    return conn