"""Step 3 of RAG: INDEX the chunks twice, densely and sparsely, in sync.

For each chunking strategy we
  1. chunk every document,
  2. embed every chunk,
  3. drop near-duplicates (cosine similarity above 0.95 to a chunk already
     kept), so retrieval does not waste its five slots on the same text,
  4. write the survivors to the vector store in one transaction.

The BM25 index is never written separately. It is rebuilt from the vector
store's own rows (retrieve/pipeline.py), so a chunk can never exist in one
index and not the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..embed import Embedder, get_embedder
from ..store import StoredChunk, VectorStore
from .chunking import STRATEGIES, chunk_document
from .loaders import Document, load_corpus


@dataclass
class IndexReport:
    strategy: str
    documents: int
    chunks: int
    duplicates_dropped: int
    stored: int


def dedupe(items: list[StoredChunk], threshold: float) -> tuple[list[StoredChunk], int]:
    """Keep the first of any group of chunks whose embeddings are more similar than `threshold`."""
    kept: list[StoredChunk] = []
    matrix: list[np.ndarray] = []
    dropped = 0
    for item in items:
        v = np.array(item.embedding, dtype=np.float32)
        v /= np.linalg.norm(v) or 1.0
        if matrix and float(np.max(np.stack(matrix) @ v)) > threshold:
            dropped += 1
            continue
        kept.append(item)
        matrix.append(v)
    return kept, dropped


def index_documents(
    docs: list[Document],
    store: VectorStore,
    strategies: list[str] | None = None,
    embedder: Embedder | None = None,
) -> list[IndexReport]:
    embedder = embedder or get_embedder()
    reports: list[IndexReport] = []
    for strategy in strategies or list(STRATEGIES):
        chunks = []
        for doc in docs:
            if strategy == "semantic":
                chunks.extend(STRATEGIES["semantic"](doc, embed=embedder.embed_passages))
            else:
                chunks.extend(chunk_document(doc, strategy))
        vectors = embedder.embed_passages([f"{c.heading}\n{c.text}" for c in chunks]) if chunks else []
        items = [StoredChunk(c, v) for c, v in zip(chunks, vectors, strict=True)]
        kept, dropped = dedupe(items, settings.dedup_threshold)
        store.replace(strategy, kept, embedder.name)
        reports.append(IndexReport(strategy, len(docs), len(chunks), dropped, store.count(strategy)))
    return reports


def index_corpus(
    store: VectorStore,
    corpus_dir: str | None = None,
    strategies: list[str] | None = None,
    embedder: Embedder | None = None,
) -> list[IndexReport]:
    return index_documents(load_corpus(corpus_dir), store, strategies, embedder)
