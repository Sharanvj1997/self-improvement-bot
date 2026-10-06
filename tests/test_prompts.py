from app.guardrails import Guardrail
from app.prompts import BASE_PROMPT, build_system_prompt


def g(rule: str, priority: str = "medium") -> Guardrail:
    return Guardrail(failure_type="x", lesson="l", guardrail=rule,
                     priority=priority, source_scenario="s")


def test_no_guardrails_returns_base_prompt():
    assert build_system_prompt([]) == BASE_PROMPT


def test_guardrails_are_injected():
    p = build_system_prompt([g("Ask for the exact date.")])
    assert "LEARNED RULES" in p and "Ask for the exact date." in p


def test_high_priority_first_and_bounded():
    gs = [g(f"rule{i}", "low") for i in range(15)] + [g("TOP", "high")]
    p = build_system_prompt(gs)
    assert "1. TOP" in p
    assert "rule10" not in p  # only the first 10 are kept


def test_base_prompt_has_no_behavioural_policies():
    # v0 must stay honest: these policies are learned, not pre-baked
    lowered = BASE_PROMPT.lower()
    assert "confirm" not in lowered and "medical" not in lowered and "ambiguous" not in lowered