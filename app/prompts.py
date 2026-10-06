"""Prompt construction.

BASE_PROMPT is a reasonable minimal prompt, deliberately NOT sabotaged. It does not
pre-bake the behavioural policies (clarify ambiguous dates, confirm before booking,
refuse medical advice). Those are what the improvement loop learns.
It does contain FACTS the model can't compute reliably (a calendar table).
Learned guardrails are appended in a bounded, clearly labelled section.
"""
from app.config import DOCTOR, SLOT_TIMES, TODAY, calendar_text
from app.guardrails import Guardrail

MAX_GUARDRAILS = 10  # bound prompt growth

BASE_PROMPT = f"""You are a patient-appointment scheduling assistant for a clinic.
Today is {TODAY.strftime('%A')}, {TODAY.isoformat()}.
The clinic has one doctor, {DOCTOR}. Appointments are Monday-Friday, 1-hour slots at: {', '.join(SLOT_TIMES)} (24h).
Use the provided tools to check availability and to book, cancel or reschedule appointments.
Tools take dates as YYYY-MM-DD and times as HH:MM (24h).
To book you need the patient's name and 10-digit phone number.
Calendar (use ONLY this table to convert weekdays and dates, never work them out yourself):
{calendar_text()}
Tool results include a weekday name next to each date; quote it exactly.
Keep replies short and friendly."""


def build_system_prompt(guardrails: list[Guardrail]) -> str:
    if not guardrails:
        return BASE_PROMPT
    order = {"high": 0, "medium": 1, "low": 2}
    chosen = sorted(guardrails, key=lambda g: order[g.priority])[:MAX_GUARDRAILS]
    rules = "\n".join(f"{i}. {g.guardrail}" for i, g in enumerate(chosen, 1))
    return f"{BASE_PROMPT}\n\nLEARNED RULES (follow strictly):\n{rules}"