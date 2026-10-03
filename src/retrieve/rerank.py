"""Rerankers: re-score the top ~20 fused candidates and keep the best 5.

Retrievers are built for recall: they look at the query and each passage
separately. A reranker reads the query and a passage TOGETHER, which is slower
but far more precise, so it only runs on a short candidate list.

* `CrossEncoderReranker` (default) - ms-marco-MiniLM-L-6-v2 run locally with
  fastembed. Free, no API key. Its raw score is a logit; we turn it into a
  0..1 relevance probability with the sigmoid function.
* `ClaudeReranker` - asks Claude Haiku to rate each passage 0-10. Needs a key.
* `LexicalReranker` - share of query terms found in the passage. Deterministic,
  used for offline tests only.

Every reranker returns scores in 0..1, so the "not in the docs" gate can use
one threshold whichever reranker is active.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol

from ..config import settings
from ..llm import LLM, extract_json
from ..store import ScoredChunk
from .sparse import tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name or settings.reranker_model
        self._model = TextCrossEncoder(model_name=self.name)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        docs = [f"{c.chunk.heading}\n{c.chunk.text}" for c in candidates]
        logits = list(self._model.rerank(query, docs))
        scored = [ScoredChunk(c.chunk, _sigmoid(float(s))) for c, s in zip(candidates, logits, strict=True)]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]


class ClaudeReranker:
    SYSTEM = (
        "You rate how well each passage answers a question about operating software systems. "
        'Reply with JSON only: {"scores": [s1, s2, ...]} with one integer 0-10 per passage, in order. '
        "10 = directly answers it, 0 = unrelated."
    )

    def __init__(self, llm: LLM | None = None):
        self.name = f"claude:{settings.rerank_model}"
        self.llm = llm or LLM()

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        passages = "\n\n".join(f"[{i}] {c.chunk.heading}\n{c.chunk.text}" for i, c in enumerate(candidates, 1))
        reply = self.llm.complete(
            system=self.SYSTEM,
            user=f"Question: {query}\n\nPassages:\n{passages}",
            model=settings.rerank_model,
            max_tokens=300,
        )
        try:
            raw = extract_json(reply)["scores"]
            scores = [max(0.0, min(10.0, float(s))) / 10.0 for s in raw]
        except (ValueError, KeyError, TypeError):
            scores = []
        if len(scores) != len(candidates):  # malformed reply: keep the fused order
            scores = [1.0 - i / len(candidates) for i in range(len(candidates))]
        scored = [ScoredChunk(c.chunk, s) for c, s in zip(candidates, scores, strict=True)]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]


class LexicalReranker:
    name = "lexical"

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        terms = set(tokenize(query))
        if not terms:
            return candidates[:top_k]
        scored = []
        for c in candidates:
            found = set(tokenize(f"{c.chunk.heading} {c.chunk.text}"))
            scored.append(ScoredChunk(c.chunk, len(terms & found) / len(terms)))
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]


@lru_cache(maxsize=4)
def _build(kind: str, model: str) -> Reranker | None:
    if kind in ("none", ""):
        return None
    if kind == "cross-encoder":
        return CrossEncoderReranker(model)
    if kind == "claude":
        return ClaudeReranker()
    if kind == "lexical":
        return LexicalReranker()
    raise ValueError(f"unknown reranker '{kind}' (cross-encoder | claude | lexical | none)")


def get_reranker() -> Reranker | None:
    return _build(settings.reranker, settings.reranker_model)
