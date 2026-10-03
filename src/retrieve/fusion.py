"""Reciprocal Rank Fusion (RRF): merge ranked lists by RANK, not by score.

Cosine similarity (0..1) and BM25 scores (0..unbounded) are on different
scales, so adding them directly lets one retriever drown out the other. RRF
ignores the raw scores and gives each document 1 / (k + rank) from every list
it appears in. A document ranked highly by both retrievers wins; one that only
one retriever found can still make the cut. k = 60 is the value from the
original RRF paper and dampens the advantage of the very top ranks.
"""

from __future__ import annotations

from ..store import ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]], k: int = 60, weights: list[float] | None = None
) -> list[ScoredChunk]:
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("one weight per ranked list")
    scores: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}
    for weight, ranked in zip(weights, ranked_lists, strict=True):
        for rank, item in enumerate(ranked, start=1):
            key = item.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            chunks.setdefault(key, item)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [ScoredChunk(chunks[key].chunk, scores[key]) for key in ordered]
