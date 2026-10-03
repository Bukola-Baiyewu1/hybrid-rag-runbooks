import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.service import set_athena


@pytest.fixture
def client(athena):
    set_athena(athena)
    with TestClient(app) as c:
        yield c
    set_athena(None)


def test_health_and_ready(client):
    assert client.get("/health").json() == {"ok": True}
    body = client.get("/ready").json()
    assert body["ok"] and body["chunks"] > 0 and body["embedder"].startswith("hashing")


def test_ask(client):
    r = client.post("/ask", json={"question": "How do I release a stale Terraform state lock?"})
    assert r.status_code == 200
    body = r.json()
    assert not body["abstained"]
    assert "force-unlock" in body["answer"]
    assert body["citations"][0]["source"] == "terraform-state-lock.md"
    assert body["verification"]["claims"]


def test_ask_abstains_on_unrelated_question(client):
    body = client.post("/ask", json={"question": "Who won the 1998 football world cup?"}).json()
    assert body["abstained"]


def test_retrieve_contract_used_by_aegis(client):
    r = client.post("/retrieve", json={"query": "5xx errors after a deploy restart", "strategy": "headers", "top_k": 3})
    body = r.json()
    assert r.status_code == 200 and len(body["results"]) == 3 and body["relevant"] is True
    first = body["results"][0]
    for key in ("chunk_id", "source", "heading", "lines", "text", "score"):
        assert key in first
    assert first["chunk_id"].startswith("high-error-rate#")


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 2000},
        {"question": "ok?", "mode": "magic"},
        {"question": "ok?", "strategy": "x"},
    ],
)
def test_ask_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_documents_lists_every_runbook(client):
    body = client.get("/documents").json()
    sources = {d["source"] for d in body["documents"]}
    assert "tls-cert-rotation.md" in sources and len(sources) == 20
    assert set(body["strategies"]) == {"fixed", "headers", "semantic"}


def test_ingest_requires_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("ATHENA_ADMIN_TOKEN", "s3cret-token")
    assert client.post("/ingest", json={}).status_code == 401
    assert client.post("/ingest", json={}, headers={"X-Admin-Token": "wrong"}).status_code == 401
    r = client.post("/ingest", json={"strategies": ["headers"]}, headers={"X-Admin-Token": "s3cret-token"})
    assert r.status_code == 200 and r.json()["indexed"][0]["strategy"] == "headers"
