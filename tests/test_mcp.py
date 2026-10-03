import asyncio

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from src.mcp_server import mcp
from src.service import set_athena


@pytest.fixture(autouse=True)
def shared(athena):
    set_athena(athena)
    yield
    set_athena(None)


def call(name, args):
    async def go():
        async with Client(mcp) as c:
            return (await c.call_tool(name, args)).data

    return asyncio.run(go())


def test_tools_are_listed():
    async def go():
        async with Client(mcp) as c:
            return {t.name for t in await c.list_tools()}

    assert asyncio.run(go()) == {"retrieve", "ask"}


def test_retrieve_tool():
    out = call("retrieve", {"query": "kubectl drain a NotReady node", "top_k": 2})
    assert len(out["results"]) == 2
    assert out["results"][0]["source"] == "node-not-ready.md"


def test_ask_tool():
    out = call("ask", {"question": "How often should database passwords be rotated?"})
    assert not out["abstained"] and "90 days" in out["answer"]


def test_bad_arguments_are_refused():
    with pytest.raises(ToolError):
        call("retrieve", {"query": "x", "mode": "magic"})
    with pytest.raises(ToolError):
        call("retrieve", {"query": "x", "top_k": 99})
