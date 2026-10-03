from src.ingest.loaders import load_corpus, load_file, Document
from src.ingest.chunking import chunk_document, chunk_corpus


def test_corpus_loads():
    docs = load_corpus("corpus")
    assert len(docs) >= 3
    assert all(d.text for d in docs)               # no empty documents
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
    long_text = ("word " * 1000).strip()           # ~5000 characters
    chunks = chunk_document(Document(long_text, "big.md"), "fixed")
    assert len(chunks) > 1                          # it actually split
    assert all(c.char_count <= 800 + 1 for c in chunks)  # within the size cap


def test_strategies_produce_chunks_for_whole_corpus():
    docs = load_corpus("corpus")
    for strategy in ("fixed", "headers"):
        chunks = chunk_corpus(docs, strategy)
        assert len(chunks) >= len(docs)             # at least one chunk per doc
        assert all(c.strategy == strategy for c in chunks)
