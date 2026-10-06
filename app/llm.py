"""Thin LLM interface. Everything provider-specific lives here.

Content blocks use a normalised dict shape:
  {"type": "text", "text": ...}
  {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    content: list[dict]

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b["type"] == "text").strip()

    @property
    def tool_calls(self) -> list[dict]:
        return [b for b in self.content if b["type"] == "tool_use"]


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class AnthropicClient:
    def __init__(self, model: str | None = None):
        from anthropic import Anthropic  # lazy import: tests don't need the SDK or a key
        from dotenv import load_dotenv

        load_dotenv()
        self._client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self._model = model or os.getenv("AGENT_MODEL", "claude-sonnet-5-5")

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            temperature=0,  # reproducible eval runs
            system=system,
            messages=messages,
            tools=tools,
        )
        blocks: list[dict] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name,
                               "input": dict(b.input)})
        return LLMResponse(content=blocks)


class OpenAICompatClient:
    """Works with Groq, Gemini (OpenAI-compatible endpoint), OpenRouter, Ollama, OpenAI."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI  # lazy import, like the Anthropic client
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._effort = os.getenv("LLM_REASONING_EFFORT")  # e.g. "low" for gpt-oss: far fewer tokens

    @staticmethod
    def _convert_messages(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            c = m["content"]
            if m["role"] == "user":
                if isinstance(c, str):
                    out.append({"role": "user", "content": c})
                else:  # list of tool_result blocks
                    for b in c:
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                    "content": b["content"]})
            else:  # assistant: list of normalized blocks
                text = "".join(b["text"] for b in c if b["type"] == "text")
                calls = [{"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                         for b in c if b["type"] == "tool_use"]
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
        return out

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        oa_tools = [{"type": "function",
                     "function": {"name": t["name"], "description": t.get("description", ""),
                                  "parameters": t["input_schema"]}} for t in tools]
        for attempt in range(6):  # exponential backoff: 2, 4, 8, 16, 32s
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=self._convert_messages(system, messages),
                    **({"tools": oa_tools} if oa_tools else {}),
                    temperature=0,
                    max_tokens=2500,
                    **({"extra_body": {"reasoning_effort": self._effort}} if self._effort else {}),
                )
                break
            except Exception as e:
                if "per day" in str(e).lower():  # daily quota: retrying can't help, fail fast
                    raise
                code = getattr(e, "status_code", None)
                transient = code in (429, 500, 502, 503, 504) or e.__class__.__name__ == "APIConnectionError"
                if not transient or attempt == 5:
                    raise
                time.sleep(2 ** (attempt + 1))
        msg = resp.choices[0].message
        blocks: list[dict] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls or []:
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.function.name,
                           "input": json.loads(tc.function.arguments or "{}")})
        return LLMResponse(content=blocks)


def make_client() -> LLMClient:
    """Pick provider from env. LLM_PROVIDER=anthropic (default) or openai (any compatible)."""
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv("LLM_PROVIDER", "anthropic") == "anthropic":
        return AnthropicClient()
    return OpenAICompatClient(
        model=os.environ["AGENT_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.getenv("LLM_BASE_URL"),
    )