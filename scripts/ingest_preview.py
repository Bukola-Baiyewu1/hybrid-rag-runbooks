"""Run me to SEE ingestion + chunking working.

    python -m scripts.ingest_preview

I load the corpus, chunk it with each strategy, print how many chunks each
produced, and show one example chunk so you can eyeball the difference.
"""
from src.ingest.loaders import load_corpus
from src.ingest.chunking import chunk_corpus, STRATEGIES


def main():
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents: {[d.source for d in docs]}\n")

    for strategy in STRATEGIES:
        chunks = chunk_corpus(docs, strategy)
        print(f"=== strategy: {strategy} -> {len(chunks)} chunks ===")
        sample = chunks[0]
        print(f"  first chunk from : {sample.source}")
        print(f"  under heading    : {sample.heading or '(none)'}")
        print(f"  size (chars)     : {sample.char_count}")
        preview = sample.text.replace("\n", " ")
        print(f"  text preview     : {preview[:120]}...\n")


if __name__ == "__main__":
    main()
