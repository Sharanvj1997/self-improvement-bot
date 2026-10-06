"""Failure analysis -> structured guardrails.

Pipeline: failed checks -> grouped by failure_type -> LLM drafts lesson/guardrail wording
-> Python validates it (strict schema + safety filters) -> stored as DATA in guardrails.json.
Metadata (failure_type, source_scenario) is assigned by code, never by the LLM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from app.guardrails import GUARDRAILS_PATH, Guardrail, load_guardrails, save_guardrails

if TYPE_CHECKING:
    from app.llm import LLMClient
    from evaluation.evaluator import CheckResult, SuiteResult
    from evaluation.runner import ScenarioRun

MAX_GUARDRAIL_CHARS = 300
MAX_LESSON_CHARS = 200

_REJECT = [
    (re.compile(r"```|<[^>]+>|\bimport\s|\bexec\(|\beval\(|__\w+__|\$\(|https?://"),
     "looks like code, markup or a URL"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{10}\b"),
     "mentions a specific date or phone number (overfit to the scenario)"),
    (re.compile(r"ignore (all|any|previous|prior)|system prompt|disregard", re.I),
     "tries to override instructions"),
]

SYSTEM = "You write precise behavioural guardrails for AI assistants. Output only JSON."

PROMPT = """A clinic appointment-scheduling assistant failed an evaluation.

Failure type: {ftype}

Evidence:
{evidence}

Write ONE general guardrail that would have prevented this failure.
- Imperative form describing what the assistant must do ("Before ..., always ...").
- General: do not mention specific names, dates, phone numbers or scenario details.
- At most 2 sentences and {glimit} characters. Plain text, no code.
- "lesson": one sentence on what went wrong (max {llimit} characters).
Reply with ONLY JSON: {{"lesson": "...", "guardrail": "...", "priority": "high|medium|low"}}"""


@dataclass
class Failure:
    failure_type: str
    scenarios: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)  # one block per failing scenario


@dataclass
class ImprovementReport:
    added: list[Guardrail] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)  # already learned, still failing
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (failure_type, reason)


def _evidence(run: "ScenarioRun", failed: "list[CheckResult]") -> str:
    lines = [f"Scenario: {run.scenario.name} - {run.scenario.description}"]
    lines += [f"Failed check: {c.type} - {c.detail}" for c in failed]
    turn = 0
    for speaker, text in run.state.transcript:
        if speaker == "patient":
            turn += 1
            lines.append(f"PATIENT: {text}")
        else:
            for ev in run.state.tool_log:
                if ev.patient_turn == turn:
                    outcome = "ok" if ev.result.get("ok") else ev.result.get("error", "error")
                    lines.append(f"TOOL: {ev.name}({ev.args}) -> {outcome}")
            lines.append(f"AGENT: {text}")
    return "\n".join(line[:300] for line in lines)



GENERIC_TYPES = {"wrong_outcome"}  # downstream symptoms: learn from them only if nothing more specific failed


def collect_failures(runs: "list[ScenarioRun]", suite: "SuiteResult") -> list[Failure]:
    groups: dict[str, Failure] = {}
    for run, res in zip(runs, suite.scenarios):
        if res.error:  # infrastructure error, not an agent failure
            continue
        by_type: dict[str, list] = {}
        for c in res.checks:
            if c.passed is False and c.type != "rubric":
                by_type.setdefault(c.failure_type, []).append(c)
        specific = {t: cs for t, cs in by_type.items() if t not in GENERIC_TYPES}
        if specific:  # a wrong final state is usually just the symptom of the specific failure
            by_type = specific
        for ftype, checks in by_type.items():
            f = groups.setdefault(ftype, Failure(ftype))
            f.scenarios.append(res.name)
            f.evidence.append(_evidence(run, checks))
    return list(groups.values())


def _reject_reason(text: str, limit: int) -> str | None:
    if not text or len(text) > limit:
        return f"empty or longer than {limit} characters"
    for pattern, why in _REJECT:
        if pattern.search(text):
            return why
    return None


def propose_guardrail(failure: Failure, llm: "LLMClient") -> tuple[Guardrail | None, str]:
    prompt = PROMPT.format(ftype=failure.failure_type, evidence="\n\n".join(failure.evidence),
                           glimit=MAX_GUARDRAIL_CHARS, llimit=MAX_LESSON_CHARS)
    reason = "no attempt made"
    for _ in range(2):  # one retry if the draft is malformed or rejected
        try:
            raw = llm.complete(SYSTEM, [{"role": "user", "content": prompt}], []).text
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            lesson, rule = str(data["lesson"]).strip(), str(data["guardrail"]).strip()
            for label, txt, lim in (("lesson", lesson, MAX_LESSON_CHARS),
                                    ("guardrail", rule, MAX_GUARDRAIL_CHARS)):
                bad = _reject_reason(txt, lim)
                if bad:
                    raise ValueError(f"{label} rejected: {bad}")
            g = Guardrail(failure_type=failure.failure_type, lesson=lesson, guardrail=rule,
                          priority=data.get("priority", "medium"),
                          source_scenario=", ".join(failure.scenarios))
            return g, "ok"
        except (ValueError, KeyError, AttributeError, ValidationError) as e:
            reason = f"{type(e).__name__}: {e}"
    return None, reason


def improve(failures: list[Failure], llm: "LLMClient", path: Path = GUARDRAILS_PATH,
            apply: bool = True) -> ImprovementReport:
    existing = load_guardrails(path)
    known = {g.failure_type for g in existing}
    report = ImprovementReport()
    for f in failures:
        if f.failure_type in known:
            report.skipped_existing.append(f.failure_type)
            continue
        g, why = propose_guardrail(f, llm)
        if g:
            report.added.append(g)
        else:
            report.rejected.append((f.failure_type, why))
    if apply and report.added:
        save_guardrails(existing + report.added, path)
    return report