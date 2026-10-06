"""Central constants. The clock is FIXED so eval runs are reproducible."""
from datetime import date,timedelta
from pathlib import Path

def calendar_text() -> str:
    """Deterministic weekday<->date table so the LLM never does calendar arithmetic."""
    out = []
    for i in range(HORIZON_DAYS):
        d = TODAY + timedelta(days=i)
        out.append(f"{d.strftime('%a')} {d.isoformat()}" + (" (today)" if i == 0 else ""))
    return "; ".join(out)


TODAY = date(2026, 10, 5)  # Monday. "tomorrow" / "next Friday" resolve against this.
HORIZON_DAYS = 30          # bookable window
DOCTOR = "Dr. Meera Rao"
SLOT_TIMES = ["09:00", "10:00", "11:00", "12:00", "14:00", "15:00", "16:00"]  # 13:00 = lunch
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "appointments.db"