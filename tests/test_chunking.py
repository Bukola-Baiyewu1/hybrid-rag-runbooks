from pathlib import Path

import pytest

from src.ingest.chunking import chunk_by_headers, chunk_corpus, chunk_document, chunk_semantic
from src.ingest.loaders import Document, load_corpus, load_file


def test_corpus_loads():
    docs = load_corpus("corpus")
    assert len(docs) >= 3
    assert all(d.text for d in docs)  # no empty documents
    assert all(d.source.endswith(".md") for d in docs)


def test_header_chunking_keeps_headings():
    doc = load_file("corpus/tls-cert-rotation.md")
    chunks = chunk_document(doc, "headers")
    headings = {c.heading for c in chunks}
    # the runbook's sections should show up as chunk headings
    assert "Steps" in headings
    assert "Rollback" in headings
    # every chunk remembers which file it came from (needed for citations)
    assert all(c.source == "tls-cert-rotation.md" for c in chunks)


def test_fixed_chunking_respects_size_and_overlap():
    long_text = ("word " * 1000).strip()  # ~5000 characters
    chunks = chunk_document(Document(long_text, "big.md"), "fixed")
    assert len(chunks) > 1  # it actually split
    assert all(c.char_count <= 800 + 1 for c in chunks)  # within the size cap
    # neighbouring chunks overlap, so text on a boundary survives whole
    assert chunks[0].text[-100:] in long_text[: len(chunks[0].text) + 200]


@pytest.mark.parametrize("strategy", ["fixed", "headers", "semantic"])
def test_strategies_produce_chunks_for_whole_corpus(strategy):
    docs = load_corpus("corpus")
    chunks = chunk_corpus(docs, strategy)
    assert len(chunks) >= len(docs)  # at least one chunk per doc
    assert all(c.strategy == strategy for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"


@pytest.mark.parametrize("strategy", ["fixed", "headers"])
def test_line_ranges_point_at_the_exact_text(strategy):
    for doc in load_corpus("corpus"):
        lines = Path("corpus", doc.source).read_text(encoding="utf-8").splitlines()
        for c in chunk_document(doc, strategy):
            cited = "\n".join(lines[c.start_line - 1 : c.end_line])
            if strategy == "headers":
                assert cited.strip() == c.text
            else:  # fixed windows may start or end mid-line
                assert c.text.split()[0] in cited and c.text.split()[-1] in cited


def test_semantic_chunks_stay_inside_their_cited_lines():
    for doc in load_corpus("corpus"):
        lines = Path("corpus", doc.source).read_text(encoding="utf-8").splitlines()
        for c in chunk_document(doc, "semantic"):
            cited = " ".join(" ".join(lines[c.start_line - 1 : c.end_line]).split())
            for word in c.text.split()[:3]:
                assert word in cited


def test_header_chunk_ids_match_the_aegis_format():
    chunks = chunk_document(load_file("corpus/tls-cert-rotation.md"), "headers")
    assert "tls-cert-rotation#rollback" in {c.chunk_id for c in chunks}
    rollback = next(c for c in chunks if c.chunk_id == "tls-cert-rotation#rollback")
    assert rollback.citation() == {
        "chunk_id": "tls-cert-rotation#rollback",
        "source": "tls-cert-rotation.md",
        "heading": "Rollback",
        "lines": f"{rollback.start_line}-{rollback.end_line}",
    }


def test_oversized_section_is_split_and_keeps_its_heading():
    body = "\n".join(f"Sentence number {i} about restarting the payment service safely." for i in range(80))
    chunks = chunk_by_headers(Document(f"# T\n\n## Big section\n{body}", "big.md"), max_size=500)
    assert len(chunks) > 1
    assert all(c.heading == "Big section" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_semantic_splits_at_a_topic_shift():
    text = (
        "# Doc\n\nRedis evicts keys when memory is full. Redis memory policy allkeys-lru evicts old keys. "
        "Redis maxmemory limits memory.\n\nTerraform state lock errors block apply. Terraform force-unlock "
        "releases a stale lock. Terraform state lives remotely."
    )
    calls = []

    def embed(texts):
        calls.append(texts)
        return [[1.0, 0.0] if "redis" in t.lower() else [0.0, 1.0] for t in texts]

    chunks = chunk_semantic(Document(text, "x.md"), embed=embed, percentile=50)
    assert any("Redis" in c.text and "Terraform" not in c.text for c in chunks)
    assert any("Terraform" in c.text and "Redis" not in c.text for c in chunks)


def test_checksum_changes_with_text():
    a = chunk_document(Document("# A\n\n## S\nrestart it", "a.md"), "headers")[0]
    b = chunk_document(Document("# A\n\n## S\nscale it", "a.md"), "headers")[0]
    assert a.checksum != b.checksum and len(a.checksum) == 16
