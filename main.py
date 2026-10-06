"""Interactive CLI:  python main.py"""
import os

from dotenv import load_dotenv

from app.agent import Agent
from app.config import DB_PATH, DOCTOR, TODAY
from app.database import fresh_seeded_db
from app.guardrails import load_guardrails
from app.llm import make_client
import sys; sys.stdout.reconfigure(encoding="utf-8")

def main() -> None:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY, or use LLM_PROVIDER=openai with LLM_API_KEY.")
    if provider != "anthropic" and not os.getenv("LLM_API_KEY"):
        raise SystemExit("LLM_API_KEY is not set. Copy .env.example to .env and add your key.")

    DB_PATH.parent.mkdir(exist_ok=True)
    conn = fresh_seeded_db(DB_PATH)  # demo starts from the same known state every time
    guardrails = load_guardrails()
    agent = Agent(make_client(), conn, guardrails)

    print(f"{DOCTOR}'s clinic | today is {TODAY:%A %Y-%m-%d} | learned guardrails: {len(guardrails)}")
    print("Type 'quit' to exit.\n")

    shown = 0
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue

        try:
            reply = agent.respond(user)
        except Exception as e:
            print(f"   [error] LLM call failed: {e}\n   Please send your message again.\n")
            continue

        # Show what ACTUALLY happened (tool calls vs. what the agent claims)
        for ev in agent.state.tool_log[shown:]:
            print(f"   [tool] {ev.name}({ev.args}) -> {ev.result}")
        shown = len(agent.state.tool_log)

        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()