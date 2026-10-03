"""Runs only when ATHENA_TEST_DATABASE_URL points at PostgreSQL with pgvector (CI does this)."""

import os

import pytest

from src.embed import HashingEmbedder
from src.ingest.chunking import chunk_document
from src.ingest.loaders import Document
from src.store import PgVectorStore, StoredChunk

URL = os.getenv("ATHENA_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL.startswith("postgresql"), reason="needs ATHENA_TEST_DATABASE_URL")


def test_round_trip_search_and_atomic_replace():
    store = PgVectorStore(URL, 384)
    emb = HashingEmbedder()
    docs = [
        Document("# A\n\n## Fix\nrestart the web service", "a.md"),
        Document("# B\n\n## Fix\nscale the queue workers", "b.md"),
    ]
    chunks = [c for d in docs for c in chunk_document(d, "headers")]
    store.replace(
        "pgtest",
        [StoredChunk(c, v) for c, v in zip(chunks, emb.embed_passages([c.text for c in chunks]), strict=True)],
        emb.name,
    )
    assert store.count("pgtest") == 2
    hits = store.search("pgtest", emb.embed_query("restart web service"), 1)
    assert hits[0].chunk.source == "a.md" and 0 < hits[0].score <= 1.0001
    loaded = store.chunks("pgtest")
    assert {c.chunk_id for c in loaded} == {"a#fix", "b#fix"}
    assert loaded[0].start_line > 0 and loaded[0].checksum
    store.replace("pgtest", [], emb.name)
    assert store.count("pgtest") == 0
    assert "pgtest" not in store.strategies()


def test_dimension_mismatch_is_reported():
    PgVectorStore(URL, 384)
    with pytest.raises(RuntimeError, match="dimensional"):
        PgVectorStore(URL, 768)
