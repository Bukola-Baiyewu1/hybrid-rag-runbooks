"""Settings for the whole project, in one place.

Every value can be overridden with an environment variable (or a line in a
local `.env` file), so the same code runs on a laptop, in Docker Compose, and in
CI without edits.
"""

from __future__ import annotations

import os

from pydantic import BaseModel


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines from .env without overriding real environment variables."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip().removeprefix("export ").strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


load_dotenv(os.getenv("ATHENA_DOTENV", ".env"))


def _env(name: str, default: str) -> str:
    """An environment variable, treating an empty value as 'not set'."""
    return os.getenv(name) or default


class Settings(BaseModel):
    # ---- corpus and chunking ------------------------------------------------
    corpus_dir: str = "corpus"
    chunk_size: int = 800  # characters, fixed-size strategy
    chunk_overlap: int = 150  # characters shared by neighbouring fixed chunks
    max_section_size: int = 1200  # header strategy: split sections bigger than this
    semantic_breakpoint_percentile: float = 80.0  # semantic strategy: split at the biggest topic shifts
    default_strategy: str = "headers"

    # ---- storage --------------------------------------------------------------
    # "postgresql+psycopg://..." for pgvector, or "memory" for an in-process store.
    database_url: str = "memory"

    # ---- embeddings -----------------------------------------------------------
    embedder: str = "fastembed"  # fastembed | hashing
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    dedup_threshold: float = 0.95  # skip a chunk this similar to one already indexed

    # ---- retrieval ------------------------------------------------------------
    candidates_per_retriever: int = 20  # dense top-k and BM25 top-k before fusion
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rerank_candidates: int = 20
    top_k: int = 5
    reranker: str = "cross-encoder"  # cross-encoder | claude | none
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # ---- generation -----------------------------------------------------------
    generator: str = "auto"  # auto | claude | extractive
    anthropic_api_key: str = ""
    generation_model: str = "claude-sonnet-5-5"
    judge_model: str = "claude-sonnet-5-5"
    rerank_model: str = "claude-haiku-4-5-20251001"
    llm_timeout_seconds: float = 60.0
    max_answer_tokens: int = 800

    # ---- answer gating ----------------------------------------------------------
    # Below these, Athena says "not in the docs" instead of answering.
    min_relevance: float = 0.15  # reranker probability of the best passage (0..1)
    min_dense_similarity: float = 0.60  # used instead when no reranker runs (cosine)
    min_confidence: float = 0.50  # final confidence after citation verification
    support_threshold: float = 0.60  # lexical verifier: share of claim terms found in the source

    def generator_mode(self) -> str:
        if self.generator == "auto":
            return "claude" if self.anthropic_api_key else "extractive"
        return self.generator


def load_settings() -> Settings:
    return Settings(
        corpus_dir=_env("ATHENA_CORPUS_DIR", "corpus"),
        chunk_size=int(_env("ATHENA_CHUNK_SIZE", "800")),
        chunk_overlap=int(_env("ATHENA_CHUNK_OVERLAP", "150")),
        max_section_size=int(_env("ATHENA_MAX_SECTION_SIZE", "1200")),
        semantic_breakpoint_percentile=float(_env("ATHENA_SEMANTIC_PERCENTILE", "80")),
        default_strategy=_env("ATHENA_STRATEGY", "headers"),
        database_url=_env("DATABASE_URL", "memory"),
        embedder=_env("ATHENA_EMBEDDER", "fastembed"),
        embedding_model=_env("ATHENA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_dim=int(_env("ATHENA_EMBEDDING_DIM", "384")),
        dedup_threshold=float(_env("ATHENA_DEDUP_THRESHOLD", "0.95")),
        candidates_per_retriever=int(_env("ATHENA_CANDIDATES", "20")),
        rrf_k=int(_env("ATHENA_RRF_K", "60")),
        dense_weight=float(_env("ATHENA_DENSE_WEIGHT", "1.0")),
        sparse_weight=float(_env("ATHENA_SPARSE_WEIGHT", "1.0")),
        rerank_candidates=int(_env("ATHENA_RERANK_CANDIDATES", "20")),
        top_k=int(_env("ATHENA_TOP_K", "5")),
        reranker=_env("ATHENA_RERANKER", "cross-encoder"),
        reranker_model=_env("ATHENA_RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        generator=_env("ATHENA_GENERATOR", "auto"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY", ""),
        generation_model=_env("ATHENA_GENERATION_MODEL", "claude-sonnet-5-5"),
        judge_model=_env("ATHENA_JUDGE_MODEL", "claude-sonnet-5-5"),
        rerank_model=_env("ATHENA_RERANK_MODEL", "claude-haiku-4-5-20251001"),
        llm_timeout_seconds=float(_env("ATHENA_LLM_TIMEOUT_SECONDS", "60")),
        max_answer_tokens=int(_env("ATHENA_MAX_ANSWER_TOKENS", "800")),
        min_relevance=float(_env("ATHENA_MIN_RELEVANCE", "0.15")),
        min_dense_similarity=float(_env("ATHENA_MIN_DENSE_SIMILARITY", "0.60")),
        min_confidence=float(_env("ATHENA_MIN_CONFIDENCE", "0.50")),
        support_threshold=float(_env("ATHENA_SUPPORT_THRESHOLD", "0.60")),
    )


settings = load_settings()

# Lesson 1 names, kept so earlier code and notes still work.
CORPUS_DIR = settings.corpus_dir
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
MAX_SECTION_SIZE = settings.max_section_size
