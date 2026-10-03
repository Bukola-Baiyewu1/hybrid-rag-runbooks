"""Sparse (keyword) search with BM25.

Dense embeddings capture meaning but blur exact tokens: a config key such as
`proxy_read_timeout`, a secret name such as `tls-web-previous`, or an error
string such as `OOMKilled`. BM25 scores documents by how often the query's exact
terms appear in them (weighted by how rare each term is), so it catches what
dense search misses.

The tokenizer keeps technical tokens whole AND splits them into parts, so
"tls-web-previous" matches a query for "tls-web-previous" exactly and also a
query that only says "previous".
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..ingest.chunking import Chunk
from ..store import ScoredChunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-/:]*[a-z0-9]|[a-z0-9]")
_STOP = frozenset(
    (
        "a an and are as at be by can do does for from how i if in is it its my no not of on or should "
        "the this to was what when where which who why with you your"
    ).split()
)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        parts = [p for p in re.split(r"[_.\-/:]", tok) if p]
        if len(parts) > 1:
            tokens.append(tok)  # the whole technical token, e.g. "tls-web-previous"
        tokens.extend(p for p in parts if p not in _STOP)
    return tokens


class BM25Index:
    """Okapi BM25 with the Lucene IDF, log(1 + (N - n + 0.5) / (n + 0.5)).

    That IDF is always positive, so a term that appears in most chunks still
    counts a little instead of going negative (a known pitfall with very small
    corpora). k1 controls term-frequency saturation; b controls length
    normalisation.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(f"{c.heading} {c.text}")) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, terms: list[str], i: int) -> float:
        tf, length = self._tf[i], self._len[i]
        total = 0.0
        for t in terms:
            f = tf.get(t, 0)
            if f:
                norm = self.k1 * (1 - self.b + self.b * length / (self._avg or 1))
                total += self._idf[t] * f * (self.k1 + 1) / (f + norm)
        return total

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        scores = [(self.score(terms, i), i) for i in range(len(self.chunks))]
        scores.sort(key=lambda s: (-s[0], s[1]))
        return [ScoredChunk(self.chunks[i], s) for s, i in scores[:k] if s > 0]
