"""Scenario runner: plays the scripted patient against a fresh agent + fresh seeded DB.

Scoring lives in evaluator.py. This file only records what happened.
"""
import argparse
import sqlite3
from dataclasses import dataclass

from app.agent import Agent
from app.database import fresh_seeded_db
from app.guardrails import Guardrail, load_guardrails
from app.llm import LLMClient, make_client
from app.state import ConversationState
from evaluation.schema import Scenario, load_scenarios

import sys; sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class ScenarioRun:
    scenario: Scenario
    state: ConversationState   # transcript + tool_log (what really happened)
    conn: sqlite3.Connection   # final DB state; the evaluator queries this
    error: str | None = None   # set if the LLM call failed; counted as a failed run


def run_scenario(scenario: Scenario, llm: LLMClient,
                 guardrails: list[Guardrail] | None = None) -> ScenarioRun:
    conn = fresh_seeded_db()  # isolated, identical seed for every scenario
    agent = Agent(llm, conn, guardrails)
    error = None
    try:
        for msg in scenario.patient_turns:
            agent.respond(msg)
    except Exception as e:  # provider outage etc.: record it, don't kill the suite
        error = f"{type(e).__name__}: {e}"
    return ScenarioRun(scenario, agent.state, conn, error)


def run_all(scenarios: list[Scenario], llm: LLMClient,
            guardrails: list[Guardrail] | None = None) -> list[ScenarioRun]:
    runs = []
    for sc in scenarios:
        print(f"running {sc.name} ...", flush=True)
        runs.append(run_scenario(sc, llm, guardrails))
    return runs


def print_run(run: ScenarioRun) -> None:
    print(f"\n=== {run.scenario.name} ===")
    turn = 0
    for speaker, text in run.state.transcript:
        if speaker == "patient":
            turn += 1
            print(f"Patient: {text}")
        else:
            for ev in run.state.tool_log:
                if ev.patient_turn == turn:
                    print(f"   [tool] {ev.name}({ev.args}) -> {ev.result}")
            print(f"Agent:   {text}")
    if run.error:
        print(f"   [run error] {run.error}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from pathlib import Path
    from evaluation.evaluator import LLMJudge, evaluate_suite, print_report, save_suite

    ap = argparse.ArgumentParser(description="Run + evaluate scenarios against the agent.")
    ap.add_argument("--only", help="run a single scenario by name")
    ap.add_argument("--verbose", action="store_true", help="print full transcripts")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judged rubric checks")
    ap.add_argument("--save", help="write results JSON to this path")
    args = ap.parse_args()

    scenarios = load_scenarios()
    if args.only:
        scenarios = [s for s in scenarios if s.name == args.only]
        if not scenarios:
            raise SystemExit(f"No scenario named {args.only!r}")

    llm = make_client()
    runs = run_all(scenarios, llm, load_guardrails())
    if args.verbose:
        for r in runs:
            print_run(r)
    suite = evaluate_suite(runs, None if args.no_judge else LLMJudge(llm))
    print_report(suite)
    if args.save:
        save_suite(suite, Path(args.save))
        
if __name__ == "__main__":
    main()