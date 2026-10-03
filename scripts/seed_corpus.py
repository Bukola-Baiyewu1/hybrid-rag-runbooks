"""Index the runbook corpus with every chunking strategy.

    python -m scripts.seed_corpus                    # all strategies
    python -m scripts.seed_corpus --strategy headers

Uses DATABASE_URL (pgvector) when set, so the API and dashboard see the result.
"""

from __future__ import annotations

import argparse

from src.athena import Athena
from src.ingest.chunking import STRATEGIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES), action="append")
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args()
    athena = Athena()
    for rep in athena.ingest(corpus_dir=args.corpus, strategies=args.strategy):
        print(
            f"{rep.strategy:9} {rep.documents} documents -> {rep.chunks} chunks, "
            f"{rep.duplicates_dropped} near-duplicates dropped, {rep.stored} stored"
        )


if __name__ == "__main__":
    main()
