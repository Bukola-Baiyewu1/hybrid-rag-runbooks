"""Where chunks and their vectors live.

`PgVectorStore` keeps them in PostgreSQL with the pgvector extension (local via
Docker Compose, or Supabase in the cloud). `MemoryVectorStore` keeps them in
process memory for tests and quick experiments. Both expose the same methods.

Each strategy's chunks are stored side by side (the `strategy` column), so the
evaluation can compare fixed, header, and semantic chunking on one database.
The BM25 index is always rebuilt from these same rows (see retrieve/sparse.py),
which is what keeps the dense and sparse indexes in sync.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import settings
from .ingest.chunking import Chunk


@dataclass
class StoredChunk:
    chunk: Chunk
    embedding: list[float]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def replace(self, strategy: str, items: list[StoredChunk], embedder: str) -> None: ...

    def chunks(self, strategy: str) -> list[Chunk]: ...

    def search(self, strategy: str, query_vector: list[float], k: int) -> list[ScoredChunk]: ...

    def count(self, strategy: str) -> int: ...

    def strategies(self) -> dict[str, dict]: ...


class MemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, list[StoredChunk]] = {}
        self._matrix: dict[str, np.ndarray] = {}
        self._embedder: dict[str, str] = {}

    def replace(self, strategy: str, items: list[StoredChunk], embedder: str) -> None:
        self._items[strategy] = list(items)
        self._matrix[strategy] = (
            np.array([i.embedding for i in items], dtype=np.float32) if items else np.zeros((0, 1), dtype=np.float32)
        )
        self._embedder[strategy] = embedder

    def chunks(self, strategy: str) -> list[Chunk]:
        return [i.chunk for i in self._items.get(strategy, [])]

    def search(self, strategy: str, query_vector: list[float], k: int) -> list[ScoredChunk]:
        items = self._items.get(strategy, [])
        if not items:
            return []
        q = np.array(query_vector, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        m = self._matrix[strategy]
        sims = m @ q / np.maximum(np.linalg.norm(m, axis=1), 1e-12)
        order = np.argsort(-sims)[:k]
        return [ScoredChunk(items[i].chunk, float(sims[i])) for i in order]

    def count(self, strategy: str) -> int:
        return len(self._items.get(strategy, []))

    def strategies(self) -> dict[str, dict]:
        return {s: {"chunks": len(v), "embedder": self._embedder[s]} for s, v in self._items.items()}


_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    strategy     text        NOT NULL,
    chunk_id     text        NOT NULL,
    source       text        NOT NULL,
    heading      text        NOT NULL,
    text         text        NOT NULL,
    chunk_index  integer     NOT NULL,
    start_line   integer     NOT NULL,
    end_line     integer     NOT NULL,
    checksum     text        NOT NULL,
    embedder     text        NOT NULL,
    embedding    vector({dim}) NOT NULL,
    indexed_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (strategy);
"""


def _to_pg_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


class PgVectorStore:
    """pgvector store. Exact cosine search: right for a focused corpus of tens to
    hundreds of documents. Add an HNSW index on `embedding` for larger corpora."""

    def __init__(self, url: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim
        self.conn = psycopg.connect(_to_pg_url(url), autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(_SCHEMA.format(dim=dim))
        existing = self.conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        ).fetchone()
        if existing and existing[0] not in (-1, dim):
            raise RuntimeError(
                f"the chunks table stores {existing[0]}-dimensional vectors but the embedder produces {dim}; "
                "drop the table or use the matching embedder"
            )

    def replace(self, strategy: str, items: list[StoredChunk], embedder: str) -> None:
        """Swap a strategy's chunks in one transaction, so readers never see a half-built index."""
        with self.conn.transaction():
            self.conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (strategy, chunk_id, source, heading, text, chunk_index, start_line,"
                    " end_line, checksum, embedder, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (
                            strategy,
                            i.chunk.chunk_id,
                            i.chunk.source,
                            i.chunk.heading,
                            i.chunk.text,
                            i.chunk.chunk_index,
                            i.chunk.start_line,
                            i.chunk.end_line,
                            i.chunk.checksum,
                            embedder,
                            np.array(i.embedding, dtype=np.float32),
                        )
                        for i in items
                    ],
                )

    @staticmethod
    def _row_to_chunk(row: tuple, strategy: str) -> Chunk:
        chunk_id, source, heading, text, chunk_index, start_line, end_line, checksum = row
        return Chunk(text, source, heading, strategy, chunk_index, len(text), start_line, end_line, chunk_id, checksum)

    _COLS = "chunk_id, source, heading, text, chunk_index, start_line, end_line, checksum"

    def chunks(self, strategy: str) -> list[Chunk]:
        rows = self.conn.execute(
            f"SELECT {self._COLS} FROM chunks WHERE strategy = %s ORDER BY source, chunk_index", (strategy,)
        ).fetchall()
        return [self._row_to_chunk(r, strategy) for r in rows]

    def search(self, strategy: str, query_vector: list[float], k: int) -> list[ScoredChunk]:
        q = np.array(query_vector, dtype=np.float32)
        rows = self.conn.execute(
            f"SELECT {self._COLS}, 1 - (embedding <=> %s) AS score FROM chunks WHERE strategy = %s"
            " ORDER BY embedding <=> %s LIMIT %s",
            (q, strategy, q, k),
        ).fetchall()
        return [ScoredChunk(self._row_to_chunk(r[:8], strategy), float(r[8])) for r in rows]

    def count(self, strategy: str) -> int:
        row = self.conn.execute("SELECT count(*) FROM chunks WHERE strategy = %s", (strategy,)).fetchone()
        return int(row[0]) if row else 0

    def strategies(self) -> dict[str, dict]:
        rows = self.conn.execute(
            "SELECT strategy, count(*), min(embedder), max(indexed_at) FROM chunks GROUP BY strategy"
        ).fetchall()
        return {r[0]: {"chunks": r[1], "embedder": r[2], "indexed_at": r[3].isoformat()} for r in rows}

    def reset(self) -> None:
        self.conn.execute("DELETE FROM chunks")


def make_store(url: str | None = None, dim: int | None = None) -> VectorStore:
    url = url or settings.database_url
    if url == "memory":
        return MemoryVectorStore()
    return PgVectorStore(url, dim or settings.embedding_dim)


def dumps_chunk(c: Chunk) -> str:
    return json.dumps(c.to_dict())
