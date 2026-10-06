from datetime import date

from app.config import HORIZON_DAYS, calendar_text
from app.prompts import BASE_PROMPT


def test_calendar_weekdays_are_correct():
    txt = calendar_text()
    assert "Mon 2026-10-05 (today)" in txt
    assert "Fri 2026-10-09" in txt and "Fri 2026-10-16" in txt
    for entry in txt.split("; "):
        wd, iso = entry.split()[:2]
        assert date.fromisoformat(iso).strftime("%a") == wd


def test_calendar_is_in_base_prompt_and_covers_the_horizon():
    assert calendar_text() in BASE_PROMPT
    assert len(calendar_text().split("; ")) == HORIZON_DAYS