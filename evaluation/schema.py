"""Scenario + check schema. Scenarios are validated data, so a typo fails at load time."""
import json
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios.json"

Category = Literal["outcome", "tools", "confirmation", "state", "safety", "communication"]
DEFAULT_CATEGORY: dict[str, str] = {
    "db_booked": "outcome",
    "db_not_booked": "outcome",
    "db_status": "outcome",
    "tool_not_called": "tools",
    "no_false_success": "state",
    "reply_not_contains": "safety",
    "rubric": "communication",
    "no_mutation_before_turn": "confirmation",
    "reply_contains_any": "safety",
    "weekday_date_consistent": "state",
}


DEFAULT_FAILURE_TYPE: dict[str, str] = {
    "db_booked": "wrong_outcome",
    "db_not_booked": "wrong_outcome",
    "db_status": "wrong_outcome",
    "tool_not_called": "unexpected_tool_call",
    "no_mutation_before_turn": "action_without_confirmation",
    "no_false_success": "unbacked_success_claim",
    "weekday_date_consistent": "weekday_date_mismatch",
    "reply_not_contains": "unsafe_medical_content",
    "reply_contains_any": "missing_required_content",
    "rubric": "rubric_failure",
}



class _Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category | None = None  # override the default category for this check
    failure_type: str | None = None  # override the default failure type for this check

    def resolved_category(self) -> str:
        return self.category or DEFAULT_CATEGORY[self.type]  # type: ignore[attr-defined]

    def resolved_failure_type(self) -> str:
        return self.failure_type or DEFAULT_FAILURE_TYPE[self.type]  # type: ignore[attr-defined]


class DbBooked(_Check):
    type: Literal["db_booked"]
    phone: str
    date: str
    time: str


class DbNotBooked(_Check):
    type: Literal["db_not_booked"]
    phone: str
    date: str
    time: str


class DbStatus(_Check):
    type: Literal["db_status"]
    appointment_id: int
    status: Literal["booked", "cancelled"]


class ToolNotCalled(_Check):
    type: Literal["tool_not_called"]
    name: str


class NoFalseSuccess(_Check):
    """Agent must not claim a booking/cancellation that the tool log does not show."""
    type: Literal["no_false_success"]


class ReplyNotContains(_Check):
    type: Literal["reply_not_contains"]
    words: list[str]


class Rubric(_Check):
    """Judged by an LLM. Scored, but never decides pass/fail on its own."""
    type: Literal["rubric"]
    criterion: str

class NoMutationBeforeTurn(_Check):
    """book/cancel/reschedule must not be called in any patient turn < `turn`."""
    type: Literal["no_mutation_before_turn"]
    turn: int


class ReplyContainsAny(_Check):
    type: Literal["reply_contains_any"]
    words: list[str]


class WeekdayDateConsistent(_Check):
    """Every 'Weekday + date' the agent states must match the real calendar."""
    type: Literal["weekday_date_consistent"]

Check = Annotated[
    Union[DbBooked, DbNotBooked, DbStatus, ToolNotCalled, NoFalseSuccess, ReplyNotContains,
          Rubric, NoMutationBeforeTurn, ReplyContainsAny, WeekdayDateConsistent],
    Field(discriminator="type"),
]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    patient_turns: list[str] = Field(min_length=1)  # scripted patient messages
    checks: list[Check] = Field(min_length=1)


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[Scenario]:
    return [Scenario(**s) for s in json.loads(path.read_text(encoding="utf-8"))]