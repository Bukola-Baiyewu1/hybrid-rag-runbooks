"""Evaluation metrics.

Relevance is judged by EVIDENCE, not by chunk id: each golden question lists
the runbook file and a short phrase that the supporting text contains. A
retrieved chunk is relevant when it comes from that file and contains that
phrase. This lets one golden set score fixed, header, and semantic chunking
fairly, even though each strategy cuts different chunks.

Retrieval metrics (no model needed):
  recall@k      share of a question's evidence found in the top k
  hit@1         the first result is relevant
  MRR           1 / rank of the first relevant result
  gate accuracy the "not in the docs" gate opens for answerable questions and
                stays shut for unanswerable ones

Answer metrics (from the full ask() pipeline):
  faithfulness       share of answer sentences supported by a cited passage
  citation accuracy  share of (sentence, citation) pairs that the passage supports
  abstention accuracy  abstains on unanswerable questions, answers the rest
  correctness        live only: Claude grades the answer against the golden answer
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..ingest.chunking import Chunk

GOLDEN_PATH = Path(__file__).with_name("golden.jsonl")


@dataclass
class GoldenItem:
    id: str
    type: str  # lookup | multi_hop | unanswerable | ambiguous
    question: str
    answer: str
    evidence: list[dict]
    match: str = "all"  # all evidence must be found (multi-hop) or any (ambiguous)

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_golden(path: Path | str = GOLDEN_PATH) -> list[GoldenItem]:
    with open(path, encoding="utf-8") as f:
        return [GoldenItem(**json.loads(line)) for line in f if line.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: Chunk, evidence: dict) -> bool:
    return chunk.source == evidence["source"] and _norm(evidence["contains"]) in _norm(f"{chunk.heading} {chunk.text}")


def relevant_anywhere(chunk: Chunk, item: GoldenItem) -> bool:
    return any(is_relevant(chunk, e) for e in item.evidence)


def recall_at_k(chunks: list[Chunk], item: GoldenItem) -> float:
    if not item.evidence:
        return 0.0
    found = [any(is_relevant(c, e) for c in chunks) for e in item.evidence]
    if item.match == "any":
        return 1.0 if any(found) else 0.0
    return sum(found) / len(found)


def hit_at_1(chunks: list[Chunk], item: GoldenItem) -> float:
    return 1.0 if chunks and relevant_anywhere(chunks[0], item) else 0.0


def reciprocal_rank(chunks: list[Chunk], item: GoldenItem) -> float:
    for rank, c in enumerate(chunks, 1):
        if relevant_anywhere(c, item):
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
