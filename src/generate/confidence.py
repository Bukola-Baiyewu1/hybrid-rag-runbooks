"""One confidence score for an answer, and the decision to abstain.

    confidence = 0.5 * retrieval relevance       (does the best passage answer the question?)
               + 0.5 * supported claim share      (did every sentence survive citation checking?)

Both parts are 0..1. Below `ATHENA_MIN_CONFIDENCE` (default 0.5) Athena returns
"I don't have enough in the docs to answer that" instead of a weak answer.
"""

from __future__ import annotations

from ..config import settings
from .verify import Verification


def confidence(relevance: float, verification: Verification) -> float:
    if not verification.claims:
        return 0.0
    return round(0.5 * max(0.0, min(1.0, relevance)) + 0.5 * verification.supported_share, 4)


def should_abstain(score: float) -> bool:
    return score < settings.min_confidence
