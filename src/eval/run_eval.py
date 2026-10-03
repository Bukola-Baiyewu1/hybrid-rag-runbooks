"""Run the evaluation and print the comparison tables.

    python -m src.eval.run_eval                 # offline: retrieval matrix + extractive answers
    python -m src.eval.run_eval --live          # adds Claude answers, Claude citation judge, correctness
    python -m src.eval.run_eval --quick         # headers strategy only, for a fast check

Results are written to eval_results/latest.json and eval_results/latest.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..athena import Athena
from ..config import settings
from ..generate.answer import ClaudeGenerator, ExtractiveGenerator
from ..generate.verify import ClaudeVerifier, LexicalVerifier
from ..ingest.chunking import STRATEGIES
from ..llm import LLM, extract_json
from ..retrieve.pipeline import MODES
from .metrics import GoldenItem, hit_at_1, load_golden, mean, recall_at_k, reciprocal_rank

RESULTS_DIR = Path("eval_results")


@dataclass
class RetrievalRow:
    strategy: str
    mode: str
    recall: float
    hit1: float
    mrr: float
    gate_accuracy: float
    context_chars: float  # average characters in the top-k passages (what the model must read)
    by_type: dict = field(default_factory=dict)


def evaluate_retrieval(athena: Athena, golden: list[GoldenItem], strategy: str, mode: str) -> RetrievalRow:
    recalls: list[float] = []
    hits: list[float] = []
    rrs: list[float] = []
    gate: list[float] = []
    context: list[float] = []
    per_type: dict[str, list[float]] = {}
    for item in golden:
        result = athena.retrieve(item.question, mode=mode, strategy=strategy)
        chunks = [r.chunk for r in result.results]
        opened = athena.retriever.passes_relevance_gate(result)
        gate.append(1.0 if opened == item.answerable else 0.0)
        if not item.answerable:
            continue
        context.append(float(sum(len(c.text) for c in chunks)))
        r = recall_at_k(chunks, item)
        recalls.append(r)
        hits.append(hit_at_1(chunks, item))
        rrs.append(reciprocal_rank(chunks, item))
        per_type.setdefault(item.type, []).append(r)
    return RetrievalRow(
        strategy,
        mode,
        mean(recalls),
        mean(hits),
        mean(rrs),
        mean(gate),
        round(mean(context)),
        {t: mean(v) for t, v in per_type.items()},
    )


GRADER = (
    "You grade answers to operations questions against a reference answer. "
    'Reply with JSON only: {"score": 1} if the answer is correct and complete, {"score": 0.5} if partly '
    'correct, {"score": 0} if wrong or missing. If the reference says the information is not in the runbooks, '
    "score 1 only if the answer declines to answer."
)


def grade(llm: LLM, item: GoldenItem, answer: str) -> float:
    reply = llm.complete(
        system=GRADER,
        user=f"Question: {item.question}\nReference answer: {item.answer}\nAnswer to grade: {answer}",
        model=settings.judge_model,
        max_tokens=50,
    )
    try:
        return float(extract_json(reply)["score"])
    except (ValueError, KeyError, TypeError):
        return 0.0


def evaluate_answers(athena: Athena, golden: list[GoldenItem], strategy: str, mode: str, live: bool) -> dict:
    faith, cite, abstain_ok, correct, latency = [], [], [], [], []
    rows = []
    for item in golden:
        ans = athena.ask(item.question, mode=mode, strategy=strategy)
        abstain_ok.append(1.0 if ans.abstained == (not item.answerable) else 0.0)
        latency.append(ans.latency_ms)
        if not ans.abstained and ans.verification:
            faith.append(ans.verification.supported_share)
            if ans.verification.citation_pairs:
                cite.append(ans.verification.citation_accuracy)
        if live:
            correct.append(grade(athena.llm, item, ans.answer))
        rows.append(
            {
                "id": item.id,
                "type": item.type,
                "abstained": ans.abstained,
                "confidence": ans.confidence,
                "answer": ans.answer,
            }
        )
    return {
        "strategy": strategy,
        "mode": mode,
        "generator": athena.generator.name,
        "verifier": athena.verifier.name,
        "faithfulness": mean(faith),
        "citation_accuracy": mean(cite),
        "abstention_accuracy": mean(abstain_ok),
        "correctness": mean(correct) if live else None,
        "answered": len(faith),
        "avg_latency_ms": round(mean(latency), 1),
        "rows": rows,
    }


SWEEP = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


def relevance_sweep(athena: Athena, golden: list[GoldenItem], strategy: str = "headers") -> dict:
    """How the "not in the docs" gate would score at different reranker thresholds.

    Shows the trade-off between declining answerable questions (false refusals)
    and answering unanswerable ones (false answers). Calibrating on the same
    68 questions is optimistic; treat it as a guide, not a guarantee.
    """
    scores = []
    for item in golden:
        result = athena.retrieve(item.question, mode="hybrid_rerank", strategy=strategy)
        scores.append((result.relevance, item.answerable))
    table = []
    for t in SWEEP:
        false_refusals = sum(1 for s, a in scores if a and s < t)
        false_answers = sum(1 for s, a in scores if not a and s >= t)
        table.append(
            {
                "threshold": t,
                "accuracy": 1 - (false_refusals + false_answers) / len(scores),
                "false_refusals": false_refusals,
                "false_answers": false_answers,
            }
        )
    return {"strategy": strategy, "current": settings.min_relevance, "table": table}


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def render_markdown(meta: dict, retrieval: list[RetrievalRow], answers: list[dict], sweep: dict | None = None) -> str:
    lines = [
        "# Athena evaluation results",
        "",
        f"- Golden questions: {meta['questions']} ({meta['by_type']})",
        f"- Embedder: `{meta['embedder']}`; reranker: `{meta['reranker']}`; top-k: {meta['top_k']}",
        f"- Answers: generator `{meta['generator']}`, verifier `{meta['verifier']}`",
        "",
        "## Retrieval (answerable questions)",
        "",
        "| Strategy | Mode | Recall@k | Hit@1 | MRR | Multi-hop recall | Gate accuracy | Context chars |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in retrieval:
        lines.append(
            f"| {r.strategy} | {r.mode} | {pct(r.recall)} | {pct(r.hit1)} | {r.mrr:.3f} | "
            f"{pct(r.by_type.get('multi_hop', 0.0))} | {pct(r.gate_accuracy)} | {r.context_chars:,.0f} |"
        )
    best = max(retrieval, key=lambda r: (r.recall, r.mrr))
    lines += ["", f"Best configuration by recall: **{best.strategy} + {best.mode}**.", ""]
    lines += [
        "## Answers (end to end)",
        "",
        "| Strategy | Mode | Faithfulness | Citation accuracy | Abstention accuracy | Correctness |",
        "|---|---|---|---|---|---|",
    ]
    for a in answers:
        lines.append(
            f"| {a['strategy']} | {a['mode']} | {pct(a['faithfulness'])} | {pct(a['citation_accuracy'])} | "
            f"{pct(a['abstention_accuracy'])} | {pct(a['correctness'])} |"
        )
    if sweep and sweep["table"]:
        lines += [
            "",
            f"## Relevance gate calibration ({sweep['strategy']} + hybrid_rerank, threshold now {sweep['current']})",
            "",
            "| Reranker threshold | Gate accuracy | Answerable questions refused | Unanswerable questions answered |",
            "|---|---|---|---|",
        ]
        for row in sweep["table"]:
            lines.append(
                f"| {row['threshold']} | {pct(row['accuracy'])} | {row['false_refusals']} | {row['false_answers']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="use Claude for answers, citation judging and grading")
    parser.add_argument("--quick", action="store_true", help="headers strategy only")
    parser.add_argument("--github", action="store_true", help="also print GitHub Actions notices")
    args = parser.parse_args(argv)

    if args.live and not settings.anthropic_api_key:
        print("--live needs ANTHROPIC_API_KEY (in the environment or .env)")
        return 2

    llm = LLM()
    generator = ClaudeGenerator(llm) if args.live else ExtractiveGenerator()
    verifier = ClaudeVerifier(llm) if args.live else LexicalVerifier()
    athena = Athena(generator=generator, verifier=verifier, llm=llm)
    golden = load_golden()
    strategies = ["headers"] if args.quick else list(STRATEGIES)

    started = time.time()
    reports = athena.ingest(strategies=strategies)
    for rep in reports:
        print(f"indexed {rep.strategy:9} {rep.stored:4} chunks ({rep.duplicates_dropped} near-duplicates dropped)")

    retrieval = [evaluate_retrieval(athena, golden, s, m) for s in strategies for m in MODES]
    answer_configs = [("headers", "dense"), ("headers", "hybrid_rerank")]
    answers = [evaluate_answers(athena, golden, s, m, args.live) for s, m in answer_configs if s in strategies]
    sweep = relevance_sweep(athena, golden) if athena.retriever.reranker is not None else None

    by_type: dict[str, int] = {}
    for g in golden:
        by_type[g.type] = by_type.get(g.type, 0) + 1
    meta = {
        "questions": len(golden),
        "by_type": by_type,
        "embedder": athena.embedder.name,
        "reranker": athena.retriever.reranker.name if athena.retriever.reranker else "none",
        "top_k": settings.top_k,
        "generator": generator.name,
        "verifier": verifier.name,
        "live": args.live,
        "seconds": round(time.time() - started, 1),
        "llm_usage": vars(llm.usage),
    }
    md = render_markdown(meta, retrieval, answers, sweep)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latest.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(
            {"meta": meta, "retrieval": [vars(r) for r in retrieval], "answers": answers, "gate_sweep": sweep},
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(md)
    if args.github:  # GitHub shows at most 10 notices per step, so group them
        for strategy in strategies:
            rows = [r for r in retrieval if r.strategy == strategy]
            parts = [
                f"{r.mode}: recall={r.recall:.3f} hit1={r.hit1:.3f} mrr={r.mrr:.3f} "
                f"multihop={r.by_type.get('multi_hop', 0):.3f} gate={r.gate_accuracy:.3f} ctx={r.context_chars:.0f}"
                for r in rows
            ]
            print(f"::notice title=retrieval {strategy}::" + " | ".join(parts))
        parts = [
            f"{a['strategy']}/{a['mode']}: faithfulness={a['faithfulness']:.3f} citation={a['citation_accuracy']:.3f} "
            f"abstention={a['abstention_accuracy']:.3f} answered={a['answered']} latency_ms={a['avg_latency_ms']}"
            for a in answers
        ]
        print("::notice title=answers::" + " | ".join(parts))
        wrong = [
            f"{row['id']}({'abstained' if row['abstained'] else 'answered'})"
            for a in answers
            if a["mode"] == "hybrid_rerank"
            for row, item in zip(a["rows"], golden, strict=True)
            if row["abstained"] == item.answerable
        ]
        print("::notice title=abstention errors (headers/hybrid_rerank)::" + (" ".join(wrong) or "none"))
        if sweep:
            print(
                "::notice title=gate sweep::"
                + " | ".join(
                    f"t={r['threshold']}: acc={r['accuracy']:.3f} refused={r['false_refusals']} "
                    f"answered={r['false_answers']}"
                    for r in sweep["table"]
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
