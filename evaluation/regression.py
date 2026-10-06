"""The self-improvement loop (one command):  python -m evaluation.regression

baseline run -> failure analysis -> structured guardrails -> re-run the SAME scenarios
-> before/after comparison -> regression check (roll back if anything got worse).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from app.guardrails import GUARDRAILS_PATH, Guardrail, load_guardrails, save_guardrails
from app.llm import LLMClient, make_client
from evaluation.evaluator import (Judge, LLMJudge, ScenarioResult, SuiteResult,
                                  evaluate_suite, print_report, save_suite)
from evaluation.improvement import ImprovementReport, collect_failures, improve
from evaluation.runner import ScenarioRun, print_run, run_all
from evaluation.schema import Scenario, load_scenarios

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
HOLDOUT_PATH = Path(__file__).resolve().parent / "holdout.json"


# ------------------------------------------------------------------ running
def run_suite(scenarios: list[Scenario], llm: LLMClient, guardrails: list[Guardrail],
              repeats: int, judge: Judge | None) -> tuple[SuiteResult, list[ScenarioRun]]:
    """Run the suite `repeats` times. A scenario passes only if it passes EVERY repeat;
    its score is the mean. Returns a representative run per scenario (first failing one)."""
    passes = []
    for k in range(repeats):
        print(f"  pass {k + 1}/{repeats}")
        runs = run_all(scenarios, llm, guardrails)
        passes.append((runs, evaluate_suite(runs, judge)))

    results, rep_runs = [], []
    for i, sc in enumerate(scenarios):
        cands = [(runs[i], suite.scenarios[i]) for runs, suite in passes]
        run, res = next(((r, s) for r, s in cands if not s.passed), cands[0])
        results.append(ScenarioResult(
            name=sc.name, checks=res.checks,
            score=round(sum(s.score for _, s in cands) / len(cands), 1),
            passed=all(s.passed for _, s in cands), error=res.error))
        rep_runs.append(run)
    return SuiteResult(results), rep_runs


# --------------------------------------------------------------- comparison
@dataclass
class Comparison:
    before_score: float
    after_score: float
    delta: float
    before_passed: int
    after_passed: int
    total: int
    fixed: list[str]
    regressions: list[str]
    still_failing: list[str]
    still_passing: list[str]


def compare(before: SuiteResult, after: SuiteResult) -> Comparison:
    b = {s.name: s.passed for s in before.scenarios}
    a = {s.name: s.passed for s in after.scenarios}
    names = list(b)
    return Comparison(
        before_score=before.score, after_score=after.score,
        delta=round(after.score - before.score, 1),
        before_passed=before.passed, after_passed=after.passed, total=before.total,
        fixed=[n for n in names if not b[n] and a[n]],
        regressions=[n for n in names if b[n] and not a[n]],
        still_failing=[n for n in names if not b[n] and not a[n]],
        still_passing=[n for n in names if b[n] and a[n]],
    )


def accept(c: Comparison) -> bool:
    """Keep an improvement only if nothing that passed now fails and the score didn't drop."""
    return not c.regressions and c.after_score >= c.before_score


def print_comparison(c: Comparison, accepted: bool) -> None:
    print("\n" + "=" * 60)
    print(f"Before: {c.before_score}%  ({c.before_passed}/{c.total} passed)")
    print(f"After:  {c.after_score}%  ({c.after_passed}/{c.total} passed)")
    print(f"Improvement: {c.delta:+.1f} percentage points")
    print(f"Fixed: {', '.join(c.fixed) or 'none'}")
    print(f"Still failing: {', '.join(c.still_failing) or 'none'}")
    prev = len(c.still_passing) + len(c.regressions)
    print(f"Previously passing scenarios: {len(c.still_passing)}/{prev} still passing")
    print(f"Regressions: {', '.join(c.regressions) or 'none'}")
    print("VERDICT: " + ("ACCEPTED - guardrails kept" if accepted
                         else "REJECTED - guardrails rolled back"))
    print("=" * 60)


# --------------------------------------------------------------------- loop
@dataclass
class LoopResult:
    before: SuiteResult
    after: SuiteResult
    comparison: Comparison
    report: ImprovementReport
    accepted: bool


def _restore(path: Path, original: str | None) -> None:
    if original is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(original, encoding="utf-8")


def _abort_on_run_errors(suite: SuiteResult, label: str, path: Path, original: str | None) -> None:
    """Provider errors (quota, outage) are not agent failures: never learn from them."""
    broken = [s for s in suite.scenarios if s.error]
    if broken:
        _restore(path, original)
        raise SystemExit(
            f"\nABORTED after {label}: {len(broken)} scenario(s) hit provider errors, e.g. "
            f"{broken[0].name}: {broken[0].error[:150]}\n"
            "Guardrails restored; results not saved. Fix the provider issue (quota/rate limit) and re-run.")


def _loop(scenarios: list[Scenario], llm: LLMClient, path: Path, repeats: int,
          judge: Judge | None, incremental: bool, results_dir: Path | None,
          original: str | None) -> LoopResult:
    if not incremental:
        save_guardrails([], path)  # baseline = the agent with NO learned rules
    start = load_guardrails(path)

    print(f"\n[1/4] BASELINE ({len(start)} learned guardrails, {repeats} pass(es))")
    before, rep_runs = run_suite(scenarios, llm, start, repeats, judge)
    _abort_on_run_errors(before, "baseline", path, original)
    print_report(before)

    failures = collect_failures(rep_runs, before)
    print(f"\n[2/4] FAILURE ANALYSIS: {len(failures)} failure type(s)")
    for f in failures:
        print(f"  - {f.failure_type}: {', '.join(f.scenarios)}")

    report = improve(failures, llm, path)
    print("\n[3/4] STRUCTURED IMPROVEMENT")
    for g in report.added:
        print(f"  + [{g.priority}] {g.failure_type}  (from: {g.source_scenario})")
        print(f"      lesson:    {g.lesson}")
        print(f"      guardrail: {g.guardrail}")
    for t in report.skipped_existing:
        print(f"  = already learned but still failing: {t}")
    for t, why in report.rejected:
        print(f"  ! draft rejected for {t}: {why}")

    if not report.added:
        _restore(path, original)
        print("\nNo new guardrails were added, so there is nothing to re-run.")
        cmp = compare(before, before)
        result = LoopResult(before, before, cmp, report, accepted=not failures)
    else:
        learned = load_guardrails(path)
        print(f"\n[4/4] RE-RUN the same scenarios with {len(learned)} guardrail(s)")
        after, after_runs = run_suite(scenarios, llm, learned, repeats, judge)
        _abort_on_run_errors(after, "re-run", path, original)
        print_report(after)
        cmp = compare(before, after)
        for run in after_runs:  # show exactly what broke, so a rollback is explainable
            if run.scenario.name in cmp.regressions:
                print("\nREGRESSION - transcript with the new guardrails:")
                print_run(run)
        accepted = accept(cmp)
        if not accepted:
            _restore(path, original)
        print_comparison(cmp, accepted)
        result = LoopResult(before, after, cmp, report, accepted)

    if results_dir:
        _save(result, results_dir)
    return result


def improvement_loop(scenarios: list[Scenario], llm: LLMClient, path: Path = GUARDRAILS_PATH,
                     repeats: int = 1, judge: Judge | None = None, incremental: bool = False,
                     results_dir: Path | None = None) -> LoopResult:
    original = path.read_text(encoding="utf-8") if path.exists() else None
    try:
        return _loop(scenarios, llm, path, repeats, judge, incremental, results_dir, original)
    except BaseException:
        _restore(path, original)  # never leave guardrails.json wiped or half-written
        raise


def _save(result: LoopResult, results_dir: Path) -> None:
    save_suite(result.before, results_dir / "before.json")
    save_suite(result.after, results_dir / "after.json")
    summary = {"accepted": result.accepted, "comparison": asdict(result.comparison),
               "guardrails_added": [g.model_dump() for g in result.report.added]}
    (results_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ holdout
def holdout_check(llm: LLMClient, learned: list[Guardrail], repeats: int,
                  judge: Judge | None, results_dir: Path | None = None) -> Comparison:
    """Unseen scenarios: never used to write guardrails. Do the learned rules generalize?"""
    scenarios = load_scenarios(HOLDOUT_PATH)
    print(f"\n[HOLDOUT] {len(scenarios)} unseen scenarios, never used to write guardrails")
    print("  -- without guardrails --")
    before, _ = run_suite(scenarios, llm, [], repeats, judge)
    print_report(before)
    print("  -- with learned guardrails --")
    after, _ = run_suite(scenarios, llm, learned, repeats, judge)
    print_report(after)
    if any(s.error for s in before.scenarios + after.scenarios):
        print("  WARNING: provider errors during holdout, so these numbers are not valid.")
    cmp = compare(before, after)
    print(f"\nHOLDOUT  Before: {cmp.before_score}% ({cmp.before_passed}/{cmp.total})   "
          f"After: {cmp.after_score}% ({cmp.after_passed}/{cmp.total})   "
          f"Regressions: {', '.join(cmp.regressions) or 'none'}")
    if results_dir:
        save_suite(before, results_dir / "holdout_before.json")
        save_suite(after, results_dir / "holdout_after.json")
    return cmp


# --------------------------------------------------------------------- main
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Evaluate, learn from failures, re-run, check regressions.")
    ap.add_argument("--repeats", type=int, default=1, help="runs per scenario (all must pass)")
    ap.add_argument("--incremental", action="store_true", help="start from guardrails on disk")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judged rubric checks")
    ap.add_argument("--no-holdout", action="store_true", help="skip the held-out generalization check")
    args = ap.parse_args()

    llm = make_client()
    judge = None if args.no_judge else LLMJudge(llm)
    result = improvement_loop(load_scenarios(), llm, repeats=args.repeats, judge=judge,
                              incremental=args.incremental, results_dir=RESULTS_DIR)
    if result.accepted and not args.no_holdout and HOLDOUT_PATH.exists():
        holdout_check(llm, load_guardrails(), args.repeats, judge, RESULTS_DIR)
    raise SystemExit(0 if result.accepted else 1)


if __name__ == "__main__":
    main()