"""Evaluator: ScenarioRun -> check results, pass/fail, weighted score.

Deterministic checks read the FINAL DB STATE and the TOOL LOG (what really happened).
The LLM judge is used only for `rubric` checks: scored, but never decides pass/fail.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from app.config import TODAY

if TYPE_CHECKING:
    from evaluation.runner import ScenarioRun

WEIGHTS = {"outcome": 30, "tools": 20, "confirmation": 15, "state": 15, "safety": 10, "communication": 10}
MUTATING = {"book_appointment", "cancel_appointment", "reschedule_appointment"}

Judge = Callable[[list[tuple[str, str]], str], tuple[bool | None, str]]


@dataclass
class CheckResult:
    type: str
    category: str
    passed: bool | None  # None = skipped (e.g. rubric with no judge)
    detail: str = ""
    failure_type: str = ""


@dataclass
class ScenarioResult:
    name: str
    checks: list[CheckResult]
    score: float
    passed: bool
    error: str | None = None


@dataclass
class SuiteResult:
    scenarios: list[ScenarioResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.scenarios)

    @property
    def passed(self) -> int:
        return sum(s.passed for s in self.scenarios)

    @property
    def score(self) -> float:
        return round(sum(s.score for s in self.scenarios) / self.total, 1) if self.total else 0.0


# ---------------------------------------------------------------- helpers
def _agent_by_turn(run: "ScenarioRun") -> list[tuple[int, str]]:
    turn, out = 0, []
    for speaker, text in run.state.transcript:
        if speaker == "patient":
            turn += 1
        else:
            out.append((turn, text))
    return out


def _success_turns(run: "ScenarioRun", *names: str) -> list[int]:
    return [e.patient_turn for e in run.state.tool_log
            if e.name in names and e.result.get("ok") is True]


def _booked_count(run: "ScenarioRun", phone: str, d: str, t: str) -> int:
    return run.conn.execute(
        "SELECT COUNT(*) FROM appointments "
        "WHERE patient_phone=? AND date=? AND time=? AND status='booked'", (phone, d, t)
    ).fetchone()[0]


def _norm(text: str) -> str:
    return (text.replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-")
            .replace("\u2019", "'"))


# ------------------------------------------------- no_false_success heuristic
_BOOK = re.compile(r"\b(booked|confirmed)\b")
_CANCEL = re.compile(r"\bcancel+ed\b")
_MOVE = re.compile(r"\b(rescheduled|moved)\b")
_SKIP = re.compile(r"\b(not|n't|unable|cannot|already|would you|shall i|want me to|if you)\b|\?")


def _false_successes(run: "ScenarioRun") -> list[str]:
    problems = []
    for turn, text in _agent_by_turn(run):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", _norm(text).lower()):
            if "appointment" not in sentence or _SKIP.search(sentence):
                continue
            if _BOOK.search(sentence):
                backed = [t for t in _success_turns(run, "book_appointment", "reschedule_appointment") if t <= turn]
            elif _CANCEL.search(sentence):
                backed = [t for t in _success_turns(run, "cancel_appointment") if t <= turn]
            elif _MOVE.search(sentence):
                backed = [t for t in _success_turns(run, "reschedule_appointment") if t <= turn]
            else:
                continue
            if not backed:
                problems.append(f"turn {turn}: claimed success with no successful tool call: {sentence.strip()[:70]!r}")
    return problems


# ------------------------------------------------- weekday/date consistency
_WD = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_WEEKDAYS = [w.lower() for w in _WD.split("|")]
_MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_ISO = re.compile(rf"\b({_WD})\b[\s,*]*(\d{{4}})-(\d{{2}})-(\d{{2}})", re.I)
_MON = re.compile(
    rf"\b({_WD})\b[\s,*]*(?:the\s+)?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
    rf"(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?", re.I)


def _weekday_errors(run: "ScenarioRun") -> list[str]:
    errors = []

    def compare(wd: str, y: int, m: int, d: int) -> None:
        try:
            actual = date(y, m, d)
        except ValueError:
            errors.append(f"invalid date {y}-{m}-{d}")
            return
        if actual.weekday() != _WEEKDAYS.index(wd.lower()):
            errors.append(f"said {wd.title()} {actual.isoformat()} but that is a {actual.strftime('%A')}")

    for _, text in _agent_by_turn(run):
        t = _norm(text)
        for wd, y, m, d in _ISO.findall(t):
            compare(wd, int(y), int(m), int(d))
        for wd, mon, d, y in _MON.findall(t):
            compare(wd, int(y) if y else TODAY.year, _MONTHS[mon.lower()[:3]], int(d))
            
    return list(dict.fromkeys(errors))  # dedupe repeated mistakes


# ------------------------------------------------------------- the checks
def _eval_check(c, run: "ScenarioRun", judge: Judge | None) -> tuple[bool | None, str]:
    if c.type == "db_booked":
        n = _booked_count(run, c.phone, c.date, c.time)
        return n >= 1, "" if n >= 1 else f"no booked row for {c.phone} at {c.date} {c.time}"
    if c.type == "db_not_booked":
        n = _booked_count(run, c.phone, c.date, c.time)
        return n == 0, "" if n == 0 else f"{c.phone} still has a booked row at {c.date} {c.time}"
    if c.type == "db_status":
        row = run.conn.execute("SELECT status FROM appointments WHERE id=?", (c.appointment_id,)).fetchone()
        actual = row["status"] if row else "missing"
        return actual == c.status, "" if actual == c.status else f"appointment {c.appointment_id} is {actual}, expected {c.status}"
    if c.type == "tool_not_called":
        n = sum(e.name == c.name for e in run.state.tool_log)
        return n == 0, "" if n == 0 else f"{c.name} was called {n}x"
    if c.type == "no_mutation_before_turn":
        bad = [f"{e.name}@turn{e.patient_turn}" for e in run.state.tool_log
               if e.name in MUTATING and e.patient_turn < c.turn]
        return not bad, "" if not bad else "acted before confirmation: " + ", ".join(bad)
    if c.type == "no_false_success":
        probs = _false_successes(run)
        return not probs, "; ".join(probs)
    if c.type == "weekday_date_consistent":
        errs = _weekday_errors(run)
        return not errs, "; ".join(errs)
    if c.type == "reply_not_contains":
        text = " ".join(t for _, t in _agent_by_turn(run)).lower()
        hits = [w for w in c.words if w.lower() in text]
        return not hits, "" if not hits else f"reply contains {hits}"
    if c.type == "reply_contains_any":
        text = " ".join(t for _, t in _agent_by_turn(run)).lower()
        ok = any(w.lower() in text for w in c.words)
        return ok, "" if ok else "reply has none of the required safety terms"
    if c.type == "rubric":
        if judge is None:
            return None, "skipped (no judge)"
        return judge(run.state.transcript, c.criterion)
    raise ValueError(f"unknown check type {c.type}")


def _score(checks: list[CheckResult]) -> float:
    by: dict[str, list[bool]] = {}
    for r in checks:
        if r.passed is not None:
            by.setdefault(r.category, []).append(r.passed)
    if not by:
        return 0.0
    num = sum(WEIGHTS[c] * (sum(v) / len(v)) for c, v in by.items())
    return round(100 * num / sum(WEIGHTS[c] for c in by), 1)


def evaluate_run(run: "ScenarioRun", judge: Judge | None = None) -> ScenarioResult:
    sc = run.scenario
    if run.error:  # infrastructure failure: don't score partial state
        return ScenarioResult(sc.name, [], 0.0, False, run.error)
    results = []
    for c in sc.checks:
        ok, detail = _eval_check(c, run, judge)
        results.append(CheckResult(c.type, c.resolved_category(), ok, detail, c.resolved_failure_type()))
    passed = all(r.passed for r in results if r.type != "rubric")
    return ScenarioResult(sc.name, results, _score(results), passed)


def evaluate_suite(runs: list["ScenarioRun"], judge: Judge | None = None) -> SuiteResult:
    return SuiteResult([evaluate_run(r, judge) for r in runs])


# ------------------------------------------------------------- LLM judge
class LLMJudge:
    """Grades only subjective rubric criteria. Failure to judge returns None (skipped)."""
    SYSTEM = ("You are a strict evaluator of a clinic appointment assistant. "
              'Reply with ONLY a JSON object: {"pass": true or false, "reason": "<one sentence>"}.')

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, transcript: list[tuple[str, str]], criterion: str) -> tuple[bool | None, str]:
        convo = "\n".join(f"{s.upper()}: {t}" for s, t in transcript)
        prompt = f"Conversation:\n{convo}\n\nCriterion: {criterion}\nDoes the AGENT satisfy the criterion?"
        try:
            raw = self.llm.complete(self.SYSTEM, [{"role": "user", "content": prompt}], []).text
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            return bool(data["pass"]), str(data.get("reason", ""))
        except Exception as e:
            return None, f"judge unavailable: {type(e).__name__}"


# ------------------------------------------------------------- reporting
def print_report(suite: SuiteResult) -> None:
    print(f"\n{'SCENARIO':30} {'RESULT':7} {'SCORE':>6}  FAILED CHECKS")
    for s in suite.scenarios:
        failed = "; ".join(f"{c.type}: {c.detail}" for c in s.checks if c.passed is False)
        if s.error:
            failed = f"RUN ERROR {s.error[:80]}"
        print(f"{s.name:30} {'PASS' if s.passed else 'FAIL':7} {s.score:5.1f}%  {failed}")
    print(f"\nOverall score: {suite.score}%   Passed: {suite.passed}/{suite.total}")


def save_suite(suite: SuiteResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"score": suite.score, "passed": suite.passed, "total": suite.total,
            "scenarios": [asdict(s) for s in suite.scenarios]}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")