"""Citation verification: does each cited passage actually support its claim?

Most RAG demos stop at "the model printed [2]". Here every sentence of the
answer is a claim, every [n] after it is a citation, and each (claim, passage)
pair is checked:

* `LexicalVerifier` (offline) - a pair is supported when at least 60% of the
  claim's content words, numbers, and technical tokens appear in the passage.
  Cheap and deterministic; it catches invented commands, values, and names,
  but not every subtle paraphrase.
* `ClaudeVerifier` (live) - Claude judges each pair as supported or not,
  reading the claim and only the passage it cites.

A claim with no citation, or whose citations all fail, is unsupported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ..config import settings
from ..llm import LLM, extract_json
from ..retrieve.pipeline import Retrieved
from ..retrieve.sparse import tokenize

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Claim:
    text: str
    citations: list[int]
    supported_citations: list[int] = field(default_factory=list)
    unsupported_citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)  # [n] with no passage n

    @property
    def supported(self) -> bool:
        return bool(self.supported_citations)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "citations": self.citations,
            "supported": self.supported,
            "supported_citations": self.supported_citations,
            "unsupported_citations": self.unsupported_citations,
            "invalid_citations": self.invalid_citations,
        }


@dataclass
class Verification:
    claims: list[Claim]
    verifier: str

    @property
    def citation_pairs(self) -> int:
        return sum(len(c.citations) for c in self.claims)

    @property
    def citation_accuracy(self) -> float:
        """Share of (claim, citation) pairs where the passage supports the claim."""
        pairs = self.citation_pairs
        return sum(len(c.supported_citations) for c in self.claims) / pairs if pairs else 0.0

    @property
    def supported_share(self) -> float:
        """Share of claims backed by at least one of their citations (faithfulness)."""
        return sum(c.supported for c in self.claims) / len(self.claims) if self.claims else 0.0

    def to_dict(self) -> dict:
        return {
            "verifier": self.verifier,
            "citation_accuracy": round(self.citation_accuracy, 4),
            "supported_share": round(self.supported_share, 4),
            "claims": [c.to_dict() for c in self.claims],
        }


def split_claims(answer: str) -> list[Claim]:
    """Split an answer into sentences, each with the [n] citations attached to it."""
    claims: list[Claim] = []
    pieces = [p for line in answer.splitlines() for p in _split_sentences(line)]
    for piece in pieces:
        cites = [int(n) for n in _CITE.findall(piece)]
        text = _CITE.sub("", piece).strip(" -*\t")
        text = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()
        if len(text) >= 3:
            claims.append(Claim(text, list(dict.fromkeys(cites))))
    return claims


def _split_sentences(line: str) -> list[str]:
    """Split one line into sentences, keeping each sentence's trailing [n] markers with it."""
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?])((?:\s*\[\d+\])*)\s+(?=[A-Z0-9`\"'(])", line):
        out.append(line[start : m.end(1)])
        start = m.end()
    out.append(line[start:])
    return [p for p in out if p.strip()]


class Verifier(Protocol):
    name: str

    def verify(self, answer: str, passages: list[Retrieved]) -> Verification: ...


def _content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 2 or t.isdigit()}


def lexical_support(claim: str, passage: str) -> float:
    terms = _content_terms(claim)
    if not terms:
        return 1.0
    found = _content_terms(passage)
    return len(terms & found) / len(terms)


class LexicalVerifier:
    name = "lexical"

    def __init__(self, threshold: float | None = None):
        self.threshold = settings.support_threshold if threshold is None else threshold

    def verify(self, answer: str, passages: list[Retrieved]) -> Verification:
        claims = split_claims(answer)
        for claim in claims:
            for n in claim.citations:
                if not 1 <= n <= len(passages):
                    claim.invalid_citations.append(n)
                    continue
                p = passages[n - 1].chunk
                ok = lexical_support(claim.text, f"{p.heading} {p.text}") >= self.threshold
                (claim.supported_citations if ok else claim.unsupported_citations).append(n)
        return Verification(claims, self.name)


class ClaudeVerifier:
    SYSTEM = (
        "You check citations. For each item, decide whether the PASSAGE alone fully supports the CLAIM "
        "(every fact, command, and value in the claim must be stated in the passage). "
        'Reply with JSON only: {"verdicts": [true, false, ...]} with one boolean per item, in order.'
    )

    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()
        self.name = f"claude:{settings.judge_model}"

    def verify(self, answer: str, passages: list[Retrieved]) -> Verification:
        claims = split_claims(answer)
        pairs: list[tuple[Claim, int]] = []
        for claim in claims:
            for n in claim.citations:
                if 1 <= n <= len(passages):
                    pairs.append((claim, n))
                else:
                    claim.invalid_citations.append(n)
        if pairs:
            items = "\n\n".join(
                f"ITEM {i}\nCLAIM: {c.text}\nPASSAGE: {passages[n - 1].chunk.text}" for i, (c, n) in enumerate(pairs, 1)
            )
            reply = self.llm.complete(system=self.SYSTEM, user=items, model=settings.judge_model, max_tokens=400)
            try:
                verdicts = [bool(v) for v in extract_json(reply)["verdicts"]]
            except (ValueError, KeyError, TypeError):
                verdicts = []
            if len(verdicts) != len(pairs):  # unusable judgement: fall back to the lexical check
                lex = LexicalVerifier()
                verdicts = [lexical_support(c.text, passages[n - 1].chunk.text) >= lex.threshold for c, n in pairs]
            for (claim, n), ok in zip(pairs, verdicts, strict=True):
                (claim.supported_citations if ok else claim.unsupported_citations).append(n)
        return Verification(claims, self.name)


def get_verifier(llm: LLM | None = None) -> Verifier:
    return ClaudeVerifier(llm) if settings.generator_mode() == "claude" else LexicalVerifier()
