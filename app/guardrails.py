"""Learned guardrail schema + loader. Guardrails are DATA (validated JSON), never code."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

GUARDRAILS_PATH = Path(__file__).resolve().parent.parent / "learned" / "guardrails.json"


class Guardrail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    failure_type: str
    lesson: str
    guardrail: str
    priority: Literal["high", "medium", "low"] = "medium"
    source_scenario: str


def load_guardrails(path: Path = GUARDRAILS_PATH) -> list[Guardrail]:
    if not path.exists():
        return []
    return [Guardrail(**g) for g in json.loads(path.read_text(encoding="utf-8") or "[]")]


def save_guardrails(guardrails: list[Guardrail], path: Path = GUARDRAILS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([g.model_dump() for g in guardrails], indent=2), encoding="utf-8")