# Architecture

## Components

| Component | File | Notes |
|---|---|---|
| Loaders | `src/ingest/loaders.py` | Markdown, text, HTML (BeautifulSoup), PDF (pypdf) |
| Chunkers | `src/ingest/chunking.py` | fixed (800 chars, 150 overlap), headers (one chunk per section, ids `file#heading-slug`), semantic (embedding breakpoints at the 80th percentile of sentence-to-sentence distance) |
| Indexer | `src/ingest/index.py` | Embeds, drops near-duplicates (cosine above 0.95), replaces a strategy's rows in one transaction |
| Vector store | `src/store.py` | PostgreSQL + pgvector (exact cosine search), or in-memory NumPy for tests |
| BM25 | `src/retrieve/sparse.py` | Okapi BM25 with the Lucene IDF; tokenizer keeps `tls-web-previous` whole and also splits it |
| Fusion | `src/retrieve/fusion.py` | Reciprocal Rank Fusion, k = 60, optional weights |
| Rerankers | `src/retrieve/rerank.py` | Cross-encoder (default), Claude Haiku, lexical (tests); all return 0..1 |
| Generators | `src/generate/answer.py` | Claude (live) or extractive (offline) |
| Citation verifier | `src/generate/verify.py` | Claude judge (live) or term overlap (offline) |
| Pipeline | `src/athena.py` | retrieve, gate, generate, verify, confidence |
| API | `src/api/main.py` | FastAPI |
| MCP | `src/mcp_server.py` | `retrieve` and `ask`, read-only |
| Dashboard | `dashboard/app.py` | Streamlit, talks to the API |
| Evaluation | `src/eval/` | Golden set, metrics, runner |

## Keeping dense and sparse in sync

The indexer only writes to the vector store. The BM25 index is built from the
store's rows on first use and rebuilt whenever the row set's checksums change.
There is no second write path, so the two indexes cannot drift.

## Why rank fusion and not score fusion

Cosine similarities lie in 0..1; BM25 scores are unbounded. Adding them lets
BM25 dominate whenever a rare token matches. RRF uses only the rank in each
list: `score = sum(weight / (60 + rank))`. A test feeds a BM25 score of 10,000
against a cosine of 0.99 to prove neither list can drown the other.

## Answer gating

```
retrieve -> best passage relevance < 0.05?            -> decline (no model call)
         -> generate; model says NOT_IN_DOCS?         -> decline
         -> verify; 0.5 * relevance + 0.5 * supported share < 0.5? -> decline, keep draft
         -> answer
```

Declining before generation saves the model call for questions the runbooks
cannot answer. Declining after verification catches a model that invents a
command or value the cited passage does not contain.

## Prompt-injection stance

Passages are wrapped in `<passage>` tags and the system prompt states they are
data, not instructions. The verifier is an independent second check: an answer
that follows an injected instruction still has to be supported by the passage
it cites, or it is withheld.

## Deployment shape

`docker compose up --build` starts pgvector (internal only), the API (models
baked into the image, runs as uid 10001, `HF_HUB_OFFLINE=1`), and the dashboard.
The API indexes the corpus on first start when the store is empty.
