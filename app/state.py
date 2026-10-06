"""Conversation state.

Two separate things on purpose:
- messages:  what the LLM sees (history).
- tool_log:  what actually happened (every tool call + real result), tagged with the
             patient turn it occurred in. The evaluator reads this, not the agent's claims.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolEvent:
    patient_turn: int      # 1-based index of the patient message that triggered it
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class ConversationState:
    messages: list[dict] = field(default_factory=list)
    tool_log: list[ToolEvent] = field(default_factory=list)
    transcript: list[tuple[str, str]] = field(default_factory=list)  # (speaker, text)
    patient_turns: int = 0