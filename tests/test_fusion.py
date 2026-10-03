from src.ingest.chunking import Chunk
from src.retrieve.fusion import reciprocal_rank_fusion
from src.store import ScoredChunk


def sc(cid, score):
    return ScoredChunk(Chunk(cid, "s.md", "", "headers", 0, 1, chunk_id=cid), score)


def test_document_in_both_lists_wins():
    dense = [sc("a", 0.9), sc("b", 0.8), sc("c", 0.7)]
    sparse = [sc("c", 40.0), sc("d", 30.0), sc("a", 1.0)]
    fused = [s.chunk.chunk_id for s in reciprocal_rank_fusion([dense, sparse])]
    assert fused[0] in ("a", "c")
    assert set(fused) == {"a", "b", "c", "d"}


def test_fusion_uses_ranks_not_raw_scores():
    """A huge BM25 score must not drown out the dense list (the classic bug)."""
    dense = [sc("a", 0.99), sc("b", 0.10)]
    sparse = [sc("b", 10_000.0), sc("a", 9_999.0)]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert abs(fused[0].score - fused[1].score) < 1e-12  # symmetric ranks -> tie


def test_rrf_score_formula():
    fused = reciprocal_rank_fusion([[sc("a", 1)], [sc("a", 1)]], k=60)
    assert abs(fused[0].score - 2 / 61) < 1e-12


def test_weights_shift_the_balance():
    dense = [sc("a", 1), sc("b", 1)]
    sparse = [sc("b", 1), sc("a", 1)]
    assert reciprocal_rank_fusion([dense, sparse], weights=[2.0, 1.0])[0].chunk.chunk_id == "a"
    assert reciprocal_rank_fusion([dense, sparse], weights=[1.0, 2.0])[0].chunk.chunk_id == "b"


def test_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
