"""The agent loop: LLM proposes -> Python validates/executes -> result goes back to LLM.
The agent never touches the DB; it only calls tools.dispatch."""
import json
import re
import sqlite3
from datetime import date

from app.guardrails import Guardrail
from app.llm import LLMClient
from app.prompts import build_system_prompt
from app.state import ConversationState, ToolEvent
from app.tools import dispatch, tool_specs

MAX_TOOL_ROUNDS = 6  # hard stop against tool-call loops
FALLBACK = "Sorry, I ran into a problem. Let me connect you with the clinic staff."

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def add_weekdays(obj):
    """Deterministic: put the weekday next to every ISO date in a tool result, so the LLM
    copies it instead of computing it (LLMs are unreliable at calendar arithmetic)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = add_weekdays(v)
            if isinstance(v, str) and k.endswith("date") and _ISO.match(v):
                try:
                    out[f"{k}_weekday"] = date.fromisoformat(v).strftime("%A")
                except ValueError:
                    pass
        return out
    if isinstance(obj, list):
        return [add_weekdays(x) for x in obj]
    return obj


class Agent:
    def __init__(self, llm: LLMClient, conn: sqlite3.Connection,
                 guardrails: list[Guardrail] | None = None):
        self.llm = llm
        self.conn = conn
        self.system = build_system_prompt(guardrails or [])
        self.tools = tool_specs()
        self.state = ConversationState()

    def respond(self, user_text: str) -> str:
        s = self.state
        s.patient_turns += 1
        s.transcript.append(("patient", user_text))
        s.messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.llm.complete(self.system, s.messages, self.tools)
            s.messages.append({"role": "assistant", "content": resp.content})

            calls = resp.tool_calls
            if not calls:  # plain text reply: this turn is done
                s.transcript.append(("agent", resp.text))
                return resp.text

            results = []
            for c in calls:
                result = add_weekdays(dispatch(self.conn, c["name"], c["input"]))
                s.tool_log.append(ToolEvent(s.patient_turns, c["name"], c["input"], result))
                results.append({"type": "tool_result", "tool_use_id": c["id"],
                                "content": json.dumps(result)})
            s.messages.append({"role": "user", "content": results})

        s.transcript.append(("agent", FALLBACK))
        return FALLBACK