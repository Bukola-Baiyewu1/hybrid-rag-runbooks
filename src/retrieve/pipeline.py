"""Hybrid retrieval: dense + sparse in parallel, fused with RRF, then reranked.

    question --> dense (pgvector) top 20 --\\
             --> sparse (BM25)    top 20 ---> RRF --> rerank top 20 --> top 5

`mode` selects how much of the pipeline runs, which is exactly what the
evaluation compares:
  dense   - vector search only
  sparse  - BM25 only
  hybrid  - dense + sparse fused with RRF
  hybrid_rerank (default) - hybrid, then the reranker
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import settings
from ..embed import Embedder, get_embedder
from ..ingest.chunking import Chunk
from ..store import ScoredChunk, VectorStore
from .fusion import reciprocal_rank_fusion
from .rerank import Reranker, get_reranker
from .sparse import BM25Index

MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


@dataclass
class Retrieved:
    chunk: Chunk
    score: float  # final ranking score (meaning depends on mode)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_similarity: float | None = None

    def to_dict(self) -> dict:
        return {
            **self.chunk.citation(),
            "text": self.chunk.text,
            "strategy": self.chunk.strategy,
            "score": round(self.score, 4),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "dense_similarity": None if self.dense_similarity is None else round(self.dense_similarity, 4),
        }


@dataclass
class RetrievalResult:
    query: str
    mode: str
    strategy: str
    results: list[Retrieved]
    relevance: float  # 0..1 estimate that the best passage answers the query
    relevance_source: str  # which signal produced `relevance`
    reranker: str | None = None
    candidates: int = 0
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "strategy": self.strategy,
            "relevance": round(self.relevance, 4),
            "relevance_source": self.relevance_source,
            "reranker": self.reranker,
            "results": [r.to_dict() for r in self.results],
        }


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder | None = None, reranker: Reranker | None = None):
        self.store = store
        self.embedder = embedder or get_embedder()
        self.reranker = reranker if reranker is not None else get_reranker()
        self._bm25: dict[str, tuple[tuple[int, str], BM25Index]] = {}

    def bm25(self, strategy: str) -> BM25Index:
        """BM25 built from the store's rows; rebuilt automatically when they change."""
        chunks = self.store.chunks(strategy)
        signature = (len(chunks), "".join(c.checksum for c in chunks))
        cached = self._bm25.get(strategy)
        if cached is None or cached[0] != signature:
            self._bm25[strategy] = (signature, BM25Index(chunks))
        return self._bm25[strategy][1]

    def retrieve(
        self, query: str, mode: str = "hybrid_rerank", strategy: str | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}'; choose from {MODES}")
        strategy = strategy or settings.default_strategy
        top_k = top_k or settings.top_k
        n = settings.candidates_per_retriever

        dense: list[ScoredChunk] = []
        sparse: list[ScoredChunk] = []
        if mode != "sparse":
            dense = self.store.search(strategy, self.embedder.embed_query(query), n)
        if mode != "dense":
            sparse = self.bm25(strategy).search(query, n)
        dense_rank = {s.chunk.chunk_id: (i, s.score) for i, s in enumerate(dense, 1)}
        sparse_rank = {s.chunk.chunk_id: i for i, s in enumerate(sparse, 1)}

        if mode == "dense":
            ranked = dense
        elif mode == "sparse":
            ranked = sparse
        else:
            ranked = reciprocal_rank_fusion(
                [dense, sparse], k=settings.rrf_k, weights=[settings.dense_weight, settings.sparse_weight]
            )

        reranker_name = None
        if mode == "hybrid_rerank" and self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked[: settings.rerank_candidates], top_k)
            reranker_name = self.reranker.name
            relevance = ranked[0].score if ranked else 0.0
            relevance_source = "reranker"
        else:
            ranked = ranked[:top_k]
            sims = [dense_rank[r.chunk.chunk_id][1] for r in ranked if r.chunk.chunk_id in dense_rank]
            if sims:
                relevance, relevance_source = max(sims), "dense_similarity"
            else:  # sparse-only: share of query terms present in the best passage
                from .rerank import LexicalReranker

                lex = LexicalReranker().rerank(query, ranked, 1)
                relevance, relevance_source = (lex[0].score if lex else 0.0), "term_coverage"

        results = [
            Retrieved(
                r.chunk,
                r.score,
                dense_rank.get(r.chunk.chunk_id, (None, None))[0],
                sparse_rank.get(r.chunk.chunk_id),
                dense_rank.get(r.chunk.chunk_id, (None, None))[1],
            )
            for r in ranked
        ]
        return RetrievalResult(
            query, mode, strategy, results, float(relevance), relevance_source, reranker_name, len(dense) + len(sparse)
        )

    def passes_relevance_gate(self, result: RetrievalResult) -> bool:
        if not result.results:
            return False
        if result.relevance_source == "reranker":
            return result.relevance >= settings.min_relevance
        if result.relevance_source == "dense_similarity":
            return result.relevance >= settings.min_dense_similarity
        return result.relevance >= 0.5
