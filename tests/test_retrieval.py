import pytest

from src.config import settings
from src.embed import HashingEmbedder
from src.ingest.index import dedupe, index_documents
from src.ingest.loaders import Document
from src.retrieve.pipeline import MODES, Retriever
from src.retrieve.rerank import ClaudeReranker, LexicalReranker
from src.retrieve.sparse import BM25Index, tokenize
from src.store import MemoryVectorStore, ScoredChunk, StoredChunk


def test_tokenizer_keeps_technical_tokens_whole_and_split():
    tokens = tokenize("Restore tls-web-previous and set proxy_read_timeout; exit code 137")
    assert "tls-web-previous" in tokens and "previous" in tokens
    assert "proxy_read_timeout" in tokens and "timeout" in tokens
    assert "137" in tokens
    assert "and" not in tokens


def test_bm25_finds_exact_config_key(athena):
    hits = athena.retriever.bm25("headers").search("proxy_read_timeout", 3)
    assert hits[0].chunk.source == "nginx-502-bad-gateway.md"


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_ranked_passages(athena, mode):
    result = athena.retrieve("How do I roll back the TLS certificate after handshake failures?", mode=mode)
    assert result.results, mode
    assert result.results[0].chunk.source == "tls-cert-rotation.md"
    assert len(result.results) <= settings.top_k
    assert 0.0 <= result.relevance <= 1.0


def test_hybrid_records_ranks_from_both_retrievers(athena):
    result = athena.retrieve("Which Redis maxmemory-policy for a cache?", mode="hybrid")
    top = result.results[0]
    assert top.chunk.source == "redis-memory.md"
    assert top.dense_rank is not None or top.sparse_rank is not None


@pytest.mark.parametrize("strategy", ["fixed", "headers", "semantic"])
def test_strategies_are_searchable_separately(athena, strategy):
    result = athena.retrieve("terraform force-unlock stale lock", strategy=strategy)
    assert result.strategy == strategy
    assert all(r.chunk.strategy == strategy for r in result.results)
    assert result.results[0].chunk.source == "terraform-state-lock.md"


def test_dense_and_sparse_indexes_are_in_sync(athena):
    for strategy in ("fixed", "headers", "semantic"):
        stored = {c.chunk_id for c in athena.store.chunks(strategy)}
        bm25 = {c.chunk_id for c in athena.retriever.bm25(strategy).chunks}
        assert stored == bm25 and stored


def test_bm25_rebuilds_when_the_store_changes():
    store = MemoryVectorStore()
    emb = HashingEmbedder()
    retriever = Retriever(store, emb, LexicalReranker())
    index_documents([Document("# A\n\n## One\nalpha bravo charlie", "a.md")], store, ["headers"], emb)
    assert retriever.bm25("headers").search("alpha", 1)
    index_documents([Document("# B\n\n## Two\ndelta echo foxtrot", "b.md")], store, ["headers"], emb)
    assert not retriever.bm25("headers").search("alpha", 1)
    assert retriever.bm25("headers").search("delta", 1)[0].chunk.source == "b.md"


def test_near_duplicate_chunks_are_dropped():
    emb = HashingEmbedder()
    docs = [
        Document("# A\n\n## Steps\nRestart the web service after a bad deploy.", "a.md"),
        Document("# B\n\n## Steps\nRestart the web service after a bad deploy.", "b.md"),
        Document("# C\n\n## Steps\nScale the queue workers when the backlog grows.", "c.md"),
    ]
    store = MemoryVectorStore()
    report = index_documents(docs, store, ["headers"], emb)[0]
    assert report.duplicates_dropped == 1 and report.stored == 2


def test_dedupe_threshold():
    items = [StoredChunk(None, [1.0, 0.0]), StoredChunk(None, [0.99, 0.05]), StoredChunk(None, [0.0, 1.0])]
    kept, dropped = dedupe(items, 0.95)
    assert dropped == 1 and len(kept) == 2


def test_unrelated_question_fails_the_relevance_gate(athena):
    result = athena.retrieve("What is the capital of France?")
    assert not athena.retriever.passes_relevance_gate(result)


def test_lexical_reranker_scores_are_probabilities(athena):
    result = athena.retrieve("kubectl rollout undo deployment", mode="hybrid_rerank")
    assert result.relevance_source == "reranker"
    assert all(0 <= r.score <= 1 for r in result.results)


def test_claude_reranker_orders_by_model_scores(fake_llm, athena):
    candidates = [ScoredChunk(r.chunk, r.score) for r in athena.retrieve("redis eviction", mode="hybrid").results[:3]]
    llm = fake_llm('{"scores": [2, 9, 5]}')
    out = ClaudeReranker(llm).rerank("redis eviction", candidates, 3)
    assert [o.chunk.chunk_id for o in out] == [
        candidates[1].chunk.chunk_id,
        candidates[2].chunk.chunk_id,
        candidates[0].chunk.chunk_id,
    ]
    assert out[0].score == 0.9


def test_claude_reranker_survives_a_malformed_reply(fake_llm, athena):
    candidates = [ScoredChunk(r.chunk, r.score) for r in athena.retrieve("redis eviction", mode="hybrid").results[:3]]
    out = ClaudeReranker(fake_llm("I think passage 2 is best")).rerank("redis eviction", candidates, 3)
    assert [o.chunk.chunk_id for o in out] == [c.chunk.chunk_id for c in candidates]


def test_unknown_mode_is_rejected(athena):
    with pytest.raises(ValueError):
        athena.retrieve("x", mode="magic")


def test_bm25_idf_stays_positive_for_common_terms():
    from src.ingest.chunking import Chunk

    chunks = [
        Chunk(f"restart service {w}", "a.md", "", "headers", i, 10, chunk_id=f"a#{i}") for i, w in enumerate("xyz")
    ]
    index = BM25Index(chunks)
    hits = index.search("restart z", 3)
    assert len(hits) == 3 and hits[0].chunk.chunk_id == "a#2"
    assert all(h.score > 0 for h in hits)
