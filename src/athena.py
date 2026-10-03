"""Athena: the whole RAG loop behind two calls, `retrieve` and `ask`.

ask(question)
  1. retrieve   hybrid search + rerank             -> top 5 passages
  2. gate       best passage relevant enough?       -> if not: "not in the docs"
  3. generate   answer only from the passages, [n]  -> draft
  4. verify     each (claim, cited passage) pair    -> supported or not
  5. confidence relevance + supported share          -> if low: "not in the docs"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import settings
from .embed import Embedder, get_embedder
from .generate.answer import NOT_IN_DOCS, Generator, get_generator
from .generate.confidence import confidence, should_abstain
from .generate.verify import Verification, Verifier, get_verifier
from .ingest.index import IndexReport, index_corpus
from .llm import LLM
from .retrieve.pipeline import RetrievalResult, Retriever
from .retrieve.rerank import Reranker
from .store import VectorStore, make_store

ABSTAIN_MESSAGE = "I don't have enough in the runbooks to answer that reliably."


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    abstain_reason: str | None
    confidence: float
    retrieval: RetrievalResult
    verification: Verification | None
    draft: str | None
    generator: str
    latency_ms: float
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "confidence": self.confidence,
            "citations": [
                {"n": i, **r.chunk.citation(), "text": r.chunk.text} for i, r in enumerate(self.retrieval.results, 1)
            ],
            "verification": self.verification.to_dict() if self.verification else None,
            "retrieval": {
                "mode": self.retrieval.mode,
                "strategy": self.retrieval.strategy,
                "relevance": round(self.retrieval.relevance, 4),
                "relevance_source": self.retrieval.relevance_source,
                "reranker": self.retrieval.reranker,
            },
            "generator": self.generator,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
        }


class Athena:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        verifier: Verifier | None = None,
        llm: LLM | None = None,
    ):
        self.llm = llm or LLM()
        self.store = store or make_store()
        self.embedder = embedder or get_embedder()
        self.retriever = Retriever(self.store, self.embedder, reranker)
        self.generator = generator or get_generator(self.llm)
        self.verifier = verifier or get_verifier(self.llm)

    def ingest(self, corpus_dir: str | None = None, strategies: list[str] | None = None) -> list[IndexReport]:
        return index_corpus(self.store, corpus_dir, strategies, self.embedder)

    def retrieve(
        self, query: str, mode: str = "hybrid_rerank", strategy: str | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        return self.retriever.retrieve(query, mode=mode, strategy=strategy, top_k=top_k)

    def ask(self, question: str, mode: str = "hybrid_rerank", strategy: str | None = None) -> Answer:
        start = time.perf_counter()
        before = (self.llm.usage.input_tokens, self.llm.usage.output_tokens, self.llm.usage.calls)
        retrieval = self.retrieve(question, mode=mode, strategy=strategy)

        def finish(answer: str, abstain_reason: str | None, conf: float, verification, draft) -> Answer:
            usage = {
                "input_tokens": self.llm.usage.input_tokens - before[0],
                "output_tokens": self.llm.usage.output_tokens - before[1],
                "llm_calls": self.llm.usage.calls - before[2],
            }
            return Answer(
                question,
                answer,
                abstain_reason is not None,
                abstain_reason,
                conf,
                retrieval,
                verification,
                draft,
                self.generator.name,
                round((time.perf_counter() - start) * 1000, 1),
                usage,
            )

        if not self.retriever.passes_relevance_gate(retrieval):
            return finish(ABSTAIN_MESSAGE, "no sufficiently relevant passage", 0.0, None, None)

        draft = self.generator.generate(question, retrieval.results)
        if draft.text.strip() == NOT_IN_DOCS:
            return finish(ABSTAIN_MESSAGE, "generator found no answer in the passages", 0.0, None, draft.text)

        verification = self.verifier.verify(draft.text, retrieval.results)
        conf = confidence(retrieval.relevance, verification)
        if should_abstain(conf):
            return finish(
                ABSTAIN_MESSAGE, f"confidence {conf} below {settings.min_confidence}", conf, verification, draft.text
            )
        return finish(draft.text, None, conf, verification, draft.text)
