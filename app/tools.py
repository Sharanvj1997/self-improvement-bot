"""Tool layer: the ONLY surface the LLM can touch.

Each tool = a strict Pydantic input model + a dispatch to the deterministic service.
Bad arguments come back as structured errors the model can read and recover from.
"""
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app import appointments as svc


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckAvailability(_Strict):
    date: str = Field(description="Calendar date, YYYY-MM-DD.")
    time: str | None = Field(default=None, description="Optional HH:MM (24h) to check one slot.")


class BookAppointment(_Strict):
    patient_name: str
    phone: str = Field(description="Patient's 10-digit phone number.")
    date: str = Field(description="YYYY-MM-DD")
    time: str = Field(description="HH:MM (24h)")


class CancelAppointment(_Strict):
    appointment_id: int
    phone: str


class RescheduleAppointment(_Strict):
    appointment_id: int
    phone: str
    new_date: str
    new_time: str


class GetPatientAppointments(_Strict):
    phone: str


_DESCRIPTIONS = {
    "check_availability": "Check open slots for a date (optionally one time). Call before offering or booking a slot.",
    "book_appointment": "Book a slot for a patient.",
    "cancel_appointment": "Cancel an existing appointment owned by the patient (id + phone).",
    "reschedule_appointment": "Move an existing appointment to a new slot atomically.",
    "get_patient_appointments": "List a patient's active appointments by phone.",
}
_MODELS = {
    "check_availability": CheckAvailability,
    "book_appointment": BookAppointment,
    "cancel_appointment": CancelAppointment,
    "reschedule_appointment": RescheduleAppointment,
    "get_patient_appointments": GetPatientAppointments,
}


def tool_specs() -> list[dict]:
    """Tool definitions (name / description / JSON schema) to send to the LLM."""
    return [
        {"name": n, "description": _DESCRIPTIONS[n], "input_schema": m.model_json_schema()}
        for n, m in _MODELS.items()
    ]


def dispatch(conn: sqlite3.Connection, name: str, args: dict[str, Any]) -> dict:
    model = _MODELS.get(name)
    if model is None:
        return {"ok": False, "error": "UNKNOWN_TOOL", "message": f"No tool named '{name}'."}
    try:
        a = model(**args)
    except ValidationError as e:
        fields = sorted({str(err["loc"][0]) for err in e.errors() if err["loc"]})
        return {"ok": False, "error": "INVALID_ARGUMENTS",
                "message": f"Bad or missing arguments: {', '.join(fields)}.", "fields": fields}
    if name == "check_availability":
        return svc.check_availability(conn, a.date, a.time)
    if name == "book_appointment":
        return svc.book_appointment(conn, a.patient_name, a.phone, a.date, a.time)
    if name == "cancel_appointment":
        return svc.cancel_appointment(conn, a.appointment_id, a.phone)
    if name == "reschedule_appointment":
        return svc.reschedule_appointment(conn, a.appointment_id, a.phone, a.new_date, a.new_time)
    return svc.get_patient_appointments(conn, a.phone)