import json
import re
from pathlib import Path

from src.eval import run_eval
from src.eval.metrics import GoldenItem, hit_at_1, is_relevant, load_golden, recall_at_k, reciprocal_rank
from src.ingest.chunking import Chunk


def ch(source, text):
    return Chunk(text, source, "", "headers", 0, len(text), chunk_id=f"{source}#x")


def test_golden_set_shape():
    golden = load_golden()
    types = {g.type for g in golden}
    assert len(golden) >= 50 and types == {"lookup", "multi_hop", "unanswerable", "ambiguous"}
    assert len({g.id for g in golden}) == len(golden)
    assert all(len(g.evidence) >= 2 for g in golden if g.type == "multi_hop")
    assert all(not g.evidence for g in golden if g.type == "unanswerable")


def test_every_evidence_phrase_exists_in_the_corpus():
    for g in load_golden():
        for e in g.evidence:
            text = re.sub(r"\s+", " ", Path("corpus", e["source"]).read_text(encoding="utf-8")).lower()
            assert e["contains"].lower() in text, (g.id, e)


def test_relevance_matching_ignores_whitespace_and_case():
    assert is_relevant(ch("a.md", "Restore the\nBackup secret"), {"source": "a.md", "contains": "the backup SECRET"})
    assert not is_relevant(ch("b.md", "Restore the backup secret"), {"source": "a.md", "contains": "backup secret"})


def test_recall_hit_and_mrr():
    item = GoldenItem(
        "q", "multi_hop", "?", "!", [{"source": "a.md", "contains": "alpha"}, {"source": "b.md", "contains": "beta"}]
    )
    chunks = [ch("c.md", "noise"), ch("a.md", "alpha here")]
    assert recall_at_k(chunks, item) == 0.5
    assert hit_at_1(chunks, item) == 0.0
    assert reciprocal_rank(chunks, item) == 0.5
    any_item = GoldenItem("q", "ambiguous", "?", "!", item.evidence, match="any")
    assert recall_at_k(chunks, any_item) == 1.0


def test_offline_eval_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_eval.settings, "corpus_dir", str(Path(__file__).resolve().parents[1] / "corpus"))
    assert run_eval.main(["--quick"]) == 0
    results = json.loads((tmp_path / "eval_results" / "latest.json").read_text())
    assert {r["mode"] for r in results["retrieval"]} == {"dense", "sparse", "hybrid", "hybrid_rerank"}
    hybrid = next(r for r in results["retrieval"] if r["mode"] == "hybrid_rerank")
    assert hybrid["recall"] > 0.8
    answers = {a["mode"]: a for a in results["answers"]}
    assert answers["hybrid_rerank"]["abstention_accuracy"] > 0.8
    assert "| headers | hybrid_rerank |" in (tmp_path / "eval_results" / "latest.md").read_text()


def test_live_eval_refuses_without_a_key():
    assert run_eval.main(["--live", "--quick"]) == 2
