"""End-to-end: retrieve -> gate -> generate -> verify -> confidence."""

from src.athena import ABSTAIN_MESSAGE, Athena
from src.config import settings
from src.generate.answer import ClaudeGenerator
from src.generate.verify import ClaudeVerifier
from src.llm import LLM


def test_answerable_question_gets_a_cited_verified_answer(athena):
    ans = athena.ask("Which Redis maxmemory-policy should a cache use?")
    assert not ans.abstained
    assert "allkeys-lru" in ans.answer and "[1]" in ans.answer
    assert ans.verification.supported_share == 1.0
    assert ans.confidence >= settings.min_confidence
    body = ans.to_dict()
    assert body["citations"][0]["source"] == "redis-memory.md"
    assert body["citations"][0]["lines"]


def test_unrelated_question_abstains_honestly(athena):
    ans = athena.ask("What is the capital of France?")
    assert ans.abstained and ans.answer == ABSTAIN_MESSAGE
    assert ans.abstain_reason == "no sufficiently relevant passage"


def test_hallucinated_answer_is_caught_and_withheld(fake_llm, athena):
    """The model invents a secret name and a destructive step; verification catches it."""
    llm = fake_llm(
        "Delete the namespace and recreate it from the secret `tls-backup-2019`. [1] "
        "Then purge every ingress controller pod. [2]",
        '{"verdicts": [false, false]}',
    )
    a = Athena(
        store=athena.store,
        embedder=athena.embedder,
        generator=ClaudeGenerator(llm),
        verifier=ClaudeVerifier(llm),
        llm=llm,
    )
    ans = a.ask("How do I roll back the TLS certificate after handshake failures?")
    assert ans.abstained
    assert ans.verification.supported_share == 0.0
    assert ans.draft.startswith("Delete the namespace")  # kept for inspection, not shown as the answer
    assert ans.usage["llm_calls"] == 2


def test_model_saying_not_in_docs_abstains(fake_llm, athena):
    llm = fake_llm("NOT_IN_DOCS")
    a = Athena(store=athena.store, embedder=athena.embedder, generator=ClaudeGenerator(llm), llm=llm)
    ans = a.ask("How do I roll back the TLS certificate?")
    assert ans.abstained and ans.abstain_reason == "generator found no answer in the passages"


def test_usage_is_counted_per_answer(fake_llm, athena):
    llm = fake_llm("Restore the backup secret `tls-web-previous` and reload the ingress. [1]", '{"verdicts": [true]}')
    a = Athena(
        store=athena.store,
        embedder=athena.embedder,
        generator=ClaudeGenerator(llm),
        verifier=ClaudeVerifier(llm),
        llm=llm,
    )
    ans = a.ask("How do I roll back the TLS certificate after handshake failures?")
    assert not ans.abstained
    assert ans.usage == {"input_tokens": 200, "output_tokens": 40, "llm_calls": 2}


def test_llm_requires_a_key():
    import pytest

    from src.llm import LLMUnavailable

    with pytest.raises(LLMUnavailable):
        LLM().complete(system="s", user="u", model="m")
