"""Embeddings: turn text into vectors so similar meanings land close together.

Two implementations share one small interface:

* `FastEmbedEmbedder` (default) runs BAAI/bge-small-en-v1.5 locally through
  ONNX Runtime. It is free, needs no API key, and produces 384-dimensional,
  normalized vectors. The model (about 130 MB) downloads once on first use.
* `HashingEmbedder` is a deterministic bag-of-words hashing trick. It has no
  understanding of meaning, but it needs no download, so the test suite and
  offline development run anywhere. Never use it for reported numbers.

Queries and passages are embedded differently on purpose: bge models expect a
short instruction in front of search queries, which `query_embed` adds.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from .config import settings

Vector = list[float]


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


def cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _normalize(v: list[float]) -> Vector:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class FastEmbedEmbedder:
    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding  # heavy import, only when used

        self.name = model_name or settings.embedding_model
        self._model = TextEmbedding(model_name=self.name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_passages(self, texts: list[str]) -> list[Vector]:
        return [_normalize([float(x) for x in v]) for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> Vector:
        return _normalize([float(x) for x in next(iter(self._model.query_embed(text)))])


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Signed feature hashing of word unigrams and bigrams (test/offline only)."""

    def __init__(self, dim: int = 384):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> Vector:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        v = [0.0] * self.dim
        for feat in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(v)

    def embed_passages(self, texts: list[str]) -> list[Vector]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)


@lru_cache(maxsize=4)
def _build(kind: str, model: str, dim: int) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder(dim)
    if kind == "fastembed":
        return FastEmbedEmbedder(model)
    raise ValueError(f"unknown embedder '{kind}' (use fastembed or hashing)")


def get_embedder() -> Embedder:
    return _build(settings.embedder, settings.embedding_model, settings.embedding_dim)
