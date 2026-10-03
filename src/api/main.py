"""HTTP API.

    POST /ask        question -> grounded answer, verified citations, confidence
    POST /retrieve   query -> ranked passages (the endpoint Aegis calls)
    POST /ingest     re-index the corpus (needs ATHENA_ADMIN_TOKEN when set)
    GET  /documents  indexed runbooks and chunk counts per strategy
    GET  /health     liveness;  GET /ready  readiness (index loaded)

Run it with:  uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import os
import secrets
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..config import settings
from ..ingest.chunking import STRATEGIES
from ..service import get_athena

Mode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]
Strategy = Literal["fixed", "headers", "semantic"]

app = FastAPI(
    title="Athena - Hybrid RAG over DevOps Runbooks",
    description="Hybrid retrieval, reranking, grounded answers with verified citations.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Mode = "hybrid_rerank"
    strategy: Strategy | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Mode = "hybrid_rerank"
    strategy: Strategy | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class IngestRequest(BaseModel):
    strategies: list[Strategy] | None = None


@app.get("/health", tags=["platform"])
def health() -> dict:
    return {"ok": True}


@app.get("/ready", tags=["platform"])
def ready() -> dict:
    athena = get_athena()
    count = athena.store.count(settings.default_strategy)
    if count == 0:
        raise HTTPException(503, "index is empty; POST /ingest first")
    return {"ok": True, "chunks": count, "embedder": athena.embedder.name, "generator": athena.generator.name}


@app.post("/ask", tags=["rag"])
def ask(body: AskRequest) -> dict:
    return get_athena().ask(body.question, mode=body.mode, strategy=body.strategy).to_dict()


@app.post("/retrieve", tags=["rag"])
def retrieve(body: RetrieveRequest) -> dict:
    athena = get_athena()
    result = athena.retrieve(body.query, mode=body.mode, strategy=body.strategy, top_k=body.top_k)
    return {**result.to_dict(), "relevant": athena.retriever.passes_relevance_gate(result)}


@app.post("/ingest", tags=["admin"])
def ingest(body: IngestRequest, x_admin_token: Annotated[str | None, Header()] = None) -> dict:
    expected = os.getenv("ATHENA_ADMIN_TOKEN", "")
    if expected and not secrets.compare_digest(x_admin_token or "", expected):
        raise HTTPException(401, "admin token required")
    reports = get_athena().ingest(strategies=list(body.strategies) if body.strategies else None)
    return {"indexed": [vars(r) for r in reports]}


@app.get("/documents", tags=["rag"])
def documents() -> dict:
    athena = get_athena()
    per_source: dict[str, dict[str, int]] = {}
    for strategy in STRATEGIES:
        for chunk in athena.store.chunks(strategy):
            per_source.setdefault(chunk.source, {})[strategy] = per_source.get(chunk.source, {}).get(strategy, 0) + 1
    return {
        "strategies": athena.store.strategies(),
        "documents": [{"source": s, "chunks": c} for s, c in sorted(per_source.items())],
    }
