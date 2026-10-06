# Design Note: Self-Improving Appointment Agent

**Scope:** a small, focused take-home, not a production system. Model: Groq free tier, `openai/gpt-oss-120b`, reasoning effort low.

## Key design choices
- **The LLM is not the source of truth.** The database is. The model only proposes tool calls; Python validates and executes them.
- **Hard rules live in code, behaviour is learned.** Code and SQLite enforce slot validity, double-booking prevention (a partial UNIQUE index) and ownership checks on cancel and reschedule. Behavioural policy (ask before acting, clarify dates) is learned as guardrails.
- **The baseline prompt is deliberately minimal**, not sabotaged. A test fails if confirmation or medical rules are added to it, so the before/after cannot be staged by accident.

## Tools and conversation state
Five narrow tools (`check_availability`, `book`, `cancel`, `reschedule`, `get_patient_appointments`) with strict Pydantic inputs. Errors come back as structured values, so the agent can recover. State keeps two separate records: `messages` (what the LLM sees) and `tool_log` (what actually happened, tagged by patient turn). The tool loop is capped at 6 rounds, then hands off to staff. Weekdays are added to every tool result in code, because the model computed them wrongly.

## Evaluation
18 scripted scenarios: 12 training, 6 held-out (never used to write guardrails). **Pass/fail is deterministic**: final DB state, tool-log trajectory (nothing changed before the patient confirms), claims vs. reality ("booked" only after a successful tool call), and weekday consistency. An LLM judge scores only the subjective medical-reply rubric and never decides pass/fail. Weights: outcome 30, tools 20, confirmation 15, state 15, safety 10, communication 10.

**Why not transcript-only:** a confident "your appointment is confirmed" can be false. In my runs the agent booked, cancelled and rebooked, and quoted wrong weekdays, while the final transcript looked fine.

## Self-improvement
Failed checks map to a `failure_type` (code assigns it, not the LLM; symptoms are folded into the specific cause). The LLM drafts a lesson and guardrail, which must pass a strict schema and filters (no code, URLs, override phrases, specific dates or phone numbers). Guardrails are stored in `learned/guardrails.json`, each traceable to its source scenarios, and injected into a bounded prompt section. The change is **rolled back automatically** if any previously passing scenario fails.

## Results (real run, `results/submitted_run/`)
| | Before | After (2 guardrails) |
|---|---|---|
| Training (12) | 83.3%, 9/12 | **100%, 12/12** |
| Held-out (6) | 55.6%, 2/6 | **93.0%, 4/6** |

**Regression:** 9/9 previously passing training scenarios still pass; no held-out regressions. An earlier run was rejected by this check and rolled back.

**Not fixed, reported honestly:** (1) `h_medical_arm_numbness`: no urgent-care wording, because no guardrail was learned (the matching training scenario happened to pass). A manual chest-pain test showed the same gap. (2) `h_ambiguous_next_tuesday`: a wrong weekday in a clarifying reply. Numbers come from single runs of a non-deterministic model.

## One thing I'd change for a real clinic
Urgent symptoms (chest pain, etc.) would be caught by a **deterministic triage gate in code** that always returns an emergency message and escalates to a human, instead of relying on the model. Also needed: real authentication (phone number is only a stand-in), audit logs, PHI protection, monitoring, EHR integration, and a deterministic output validator for dates.

## Where AI helped
Claude acted as a pair-programming partner: first drafts of most code, scenarios and docs, and brainstorming the architecture and evaluation design. It also helped debug provider and quota errors. At run time, an LLM drafts the guardrail wording.

## Where my judgment overrode AI
- I set the constraints up front (SQLite, no RAG, no frontend, no heavy infrastructure) and required step-by-step building, rejecting an all-at-once code dump.
- I chose a free provider because of cost, which meant a weaker model and an adapter to support it.
- [CONFIRM or DELETE: I chose to report the two remaining held-out failures instead of tuning them away.]
- [ADD any other real decision where you disagreed with a suggestion.]