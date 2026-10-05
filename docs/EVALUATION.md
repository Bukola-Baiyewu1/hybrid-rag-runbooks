# Evaluation

All numbers come from `python -m src.eval.run_eval`, run by the CI job
"Evaluation with the real models" on every push. Models: `BAAI/bge-small-en-v1.5`
(embeddings) and `Xenova/ms-marco-MiniLM-L-6-v2` (cross-encoder reranker), both
run locally with fastembed. Top-k is 5.

## Golden set

`src/eval/golden.jsonl`, 68 hand-written questions over the 20 runbooks:

| Type | Count | What it tests |
|---|---|---|
| lookup | 43 | One fact from one section |
| multi_hop | 10 | The answer needs two sections, often from two files |
| unanswerable | 10 | The runbooks do not contain the answer; the right response is to decline |
| ambiguous | 5 | Vague questions with several acceptable sources |

Each answerable question lists its evidence as (file, phrase the supporting text
contains). A retrieved chunk is relevant when it comes from that file and
contains that phrase. Judging by evidence rather than chunk id is what makes the
three chunking strategies comparable. A test checks that every evidence phrase
really exists in the corpus.

## Retrieval: every strategy and mode

Answerable questions only (58). "Gate" is the accuracy of the decision to answer
or decline, over all 68 questions.

| Strategy | Mode | Recall@5 | Hit@1 | MRR | Multi-hop recall | Gate | Context chars |
|---|---|---|---|---|---|---|---|
| fixed | dense | 99.1% | 87.9% | 0.937 | 95.0% | 88.2% | 3,064 |
| fixed | sparse | 100.0% | 96.6% | 0.983 | 100.0% | 89.7% | 2,894 |
| fixed | hybrid | 100.0% | 93.1% | 0.966 | 100.0% | 88.2% | 3,145 |
| fixed | hybrid_rerank | 100.0% | 94.8% | 0.971 | 100.0% | 97.1% | 3,090 |
| headers | dense | 87.9% | 77.6% | 0.838 | 60.0% | 88.2% | 1,114 |
| headers | sparse | 95.7% | 75.9% | 0.853 | 85.0% | 77.9% | 1,095 |
| headers | hybrid | 94.0% | 81.0% | 0.883 | 75.0% | 88.2% | 1,116 |
| **headers** | **hybrid_rerank** | **95.7%** | **84.5%** | **0.914** | **85.0%** | **98.5%** | **1,145** |
| semantic | dense | 88.8% | 75.9% | 0.839 | 55.0% | 88.2% | 979 |
| semantic | sparse | 94.8% | 74.1% | 0.845 | 80.0% | 75.0% | 983 |
| semantic | hybrid | 92.2% | 79.3% | 0.874 | 65.0% | 88.2% | 1,019 |
| semantic | hybrid_rerank | 94.8% | 84.5% | 0.911 | 80.0% | 98.5% | 1,040 |

Findings:

1. **Reranking matters most.** Within every strategy, hybrid + rerank has the
   best Hit@1 and MRR of the fused modes and the best gate.
2. **Fusion alone is not free.** RRF of dense and BM25 lost recall against BM25
   alone in the header and semantic strategies: dense mistakes are fused in.
   The reranker recovers it.
3. **BM25 carries multi-hop questions** on this corpus (85% vs 60% for dense with
   headers), because multi-hop questions name specific tokens from both sections.
4. **Fixed windows win recall by reading more.** 800-character windows cover most
   of a short runbook (27 chunks for 20 files), so they send about 2.7 times
   more text and cite whole documents. Header sections are the default for
   precise citations at a third of the context. Semantic chunking is close to
   headers here because these runbooks are already well sectioned.

## The "not in the docs" gate

With a reranker, the gate uses the best passage's relevance probability.
Without one, it uses dense cosine similarity (threshold 0.60). Sweep for
headers + hybrid_rerank:

| Reranker threshold | Gate accuracy | Answerable questions refused | Unanswerable questions answered |
|---|---|---|---|
| 0.01 | 98.5% | 1 | 0 |
| 0.02 | 98.5% | 1 | 0 |
| **0.05 (default)** | **98.5%** | **1** | **0** |
| 0.10 | 98.5% | 1 | 0 |
| 0.15 | 97.1% | 2 | 0 |
| 0.20 | 95.6% | 3 | 0 |
| 0.30 | 94.1% | 4 | 0 |
| 0.50 | 91.2% | 6 | 0 |

Every unanswerable question scores below 0.01, so the threshold has a wide
margin on that side. The one refused answerable question is q01 ("What error
rate threshold defines a high error rate incident?"), for which the cross-encoder
scores every retrieved passage below 0.01. The threshold was chosen on this
same set, so treat 98.5% as optimistic.

## Answers end to end (headers)

| Mode | Faithfulness | Citation accuracy | Abstention accuracy | Answered |
|---|---|---|---|---|
| dense | 100% | 100% | 89.7% | 65 of 68 |
| hybrid_rerank | 100% | 100% | 98.5% | 57 of 68 |

These are offline numbers: extractive answers checked by word overlap, so
faithfulness and citation accuracy are 100% by construction. They confirm that
the pipeline cites and gates correctly; they say nothing about Claude's
writing. The live run replaces both parts with Claude:

```bash
python -m src.eval.run_eval --live --quick
```

### Live results (Claude `claude-sonnet-5-5`, October 2026)

Claude writes the answers, judges each sentence against the passage it cites,
and grades each answer against the golden answer (1 correct and complete, 0.5
partly correct, 0 wrong or missing; declining an answerable question scores 0).

| Mode | Faithfulness | Citation accuracy | Abstention accuracy | Correctness |
|---|---|---|---|---|
| dense | 93.9% | 96.0% | 91.2% | 64.7% |
| hybrid_rerank | 94.2% | 95.6% | 95.6% | 70.6% |

Retrieval and gate calibration in the live run were identical to the offline
run (Recall@5 95.7%, Hit@1 84.5%, MRR 0.914 for hybrid + rerank), as expected:
retrieval does not use Claude.

Reading the live numbers:

* Better retrieval gives better answers from the same model: hybrid + rerank is
  5.9 points more correct and 4.4 points better at abstaining than dense alone.
* Abstention accuracy is 95.6% live against 98.5% offline. The difference is the
  confidence gate: when the citation judge does not support enough of a Claude
  answer, confidence falls below 0.5 and Athena declines. Those are answerable
  questions declined, the safe direction of error.
* Correctness of 70.6% is the main thing to improve. Next steps: grade partial
  answers per missing step to see where they fall short, and test a larger
  top-k for the multi-hop questions, where recall is lowest.

## Aegis with Athena

The CI job "Aegis agent with Athena as its retriever" starts Athena over Aegis's
5 runbooks and runs Aegis's own retrieval evaluation (16 queries) with both
retrievers, then runs one full Aegis plan through Athena.

| Metric | Aegis TF-IDF | Athena |
|---|---|---|
| Correct runbook first | 93% | 86% |
| Supporting section in top 3 | 93% | 93% |
| MRR | 0.88 | 0.86 |
| Unrelated queries rejected | 100% | 100% |

Athena declines "volume almost full, delete old files?" (cross-encoder relevance
below 0.01 for every passage), and both retrievers rank the high-error-rate
section first for "memory saturated, service slow under load". Aegis keeps
TF-IDF as its default retriever.
