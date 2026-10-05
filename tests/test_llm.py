"""The Claude wrapper, with the SDK replaced by a fake client."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from src.llm import LLM


def _reply(text):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _bad_request(message):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(400, request=req)
    return anthropic.BadRequestError(message, response=resp, body={"error": {"message": message}})


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _llm(*script):
    return LLM(client=SimpleNamespace(messages=FakeMessages(script)))


def test_returns_only_text_blocks_and_counts_usage():
    llm = _llm(_reply("hello"))
    assert llm.complete(system="s", user="u", model="m") == "hello"
    assert llm.usage.calls == 1 and llm.usage.input_tokens == 10
    assert "temperature" not in llm.client.messages.calls[0]


def test_every_argument_is_accepted_by_the_installed_sdk():
    """Guards against SDK upgrades that drop a parameter (the fake client would not notice)."""
    import inspect

    accepted = inspect.signature(anthropic.resources.messages.Messages.create).parameters
    sent = LLM.request(system="s", user="u", model="m", max_tokens=10)
    assert set(sent) <= set(accepted), set(sent) - set(accepted)


def test_api_errors_propagate():
    llm = _llm(_bad_request("model: no-such-model"))
    with pytest.raises(anthropic.BadRequestError):
        llm.complete(system="s", user="u", model="m")
    assert len(llm.client.messages.calls) == 1
