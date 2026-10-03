"""Test setup: deterministic, offline, no API keys, no model downloads.

Environment variables are set BEFORE `src` is imported. Dense search uses the
hashing embedder and reranking uses the lexical reranker; the real models are
exercised by the evaluation job in CI. Set ATHENA_TEST_DATABASE_URL to run the
same tests against PostgreSQL + pgvector.
"""

import os
from types import SimpleNamespace

os.environ.update(
    {
        "ATHENA_DOTENV": "/nonexistent/.env",
        "ATHENA_EMBEDDER": "hashing",
        "ATHENA_RERANKER": "lexical",
        "ATHENA_GENERATOR": "extractive",
        "ANTHROPIC_API_KEY": "",
        "ATHENA_MIN_DENSE_SIMILARITY": "0.2",
        "ATHENA_ADMIN_TOKEN": "",
        "DATABASE_URL": os.getenv("ATHENA_TEST_DATABASE_URL", "memory"),
    }
)

import pytest  # noqa: E402

from src.athena import Athena  # noqa: E402
from src.config import settings  # noqa: E402


@pytest.fixture(scope="session")
def athena():
    a = Athena()
    a.ingest()
    return a


@pytest.fixture(autouse=True)
def restore_settings():
    saved = settings.model_dump()
    yield
    for k, v in saved.items():
        setattr(settings, k, v)


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=reply)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


class FakeClient:
    def __init__(self, *replies):
        self.messages = FakeMessages(replies)


@pytest.fixture
def fake_llm():
    from src.llm import LLM

    def make(*replies):
        return LLM(client=FakeClient(*replies))

    return make
