"""A small wrapper around the Anthropic Messages API.

Everything that talks to Claude (generation, reranking, citation judging,
answer grading) goes through `LLM.complete`, so tests can swap in a fake client
and the rest of the code never handles SDK details.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import settings


class LLMUnavailable(RuntimeError):
    """Raised when a live model call is needed but no API key is configured."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


@dataclass
class LLM:
    client: Any = None
    usage: Usage = field(default_factory=Usage)
    no_temperature: set[str] = field(default_factory=set)

    def _client(self) -> Any:
        if self.client is None:
            if not settings.anthropic_api_key:
                raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
            import anthropic

            self.client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_seconds, max_retries=3
            )
        return self.client

    def complete(self, *, system: str, user: str, model: str, max_tokens: int = 800) -> str:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if model not in self.no_temperature:
            request["temperature"] = 0
        try:
            resp = self._client().messages.create(**request)
        except Exception as exc:
            # Models that reason before answering reject a custom temperature with a 400.
            # Remember that for this model and retry once with its default sampling.
            if "temperature" not in request or getattr(exc, "status_code", None) != 400:
                raise
            if "temperature" not in str(exc).lower():
                raise
            self.no_temperature.add(model)
            request.pop("temperature")
            resp = self._client().messages.create(**request)
        u = getattr(resp, "usage", None)
        self.usage.add(Usage(int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0), 1))
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in a model reply (tolerates code fences and prose)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in model reply")
