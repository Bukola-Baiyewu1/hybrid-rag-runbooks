"""MCP server: Athena's retrieval and Q&A as tools for any MCP client
(Claude Code, Claude Desktop, Cursor, or the Aegis agent).

Run it with:  python -m src.mcp_server        (stdio transport)

Both tools are read-only. Logs go to stderr so they never mix with the MCP
protocol on stdout.
"""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .service import get_athena

mcp = FastMCP(
    "athena-rag",
    instructions="Search DevOps runbooks (retrieve) or get a cited, verified answer from them (ask).",
)

_MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


@mcp.tool(annotations={"readOnlyHint": True})
def retrieve(query: str, mode: str = "hybrid_rerank", top_k: int = 5) -> dict:
    """Return the most relevant runbook passages for a query, each with source, heading, and line range."""
    if mode not in _MODES:
        raise ToolError(f"mode must be one of {_MODES}")
    if not 1 <= top_k <= 20:
        raise ToolError("top_k must be between 1 and 20")
    athena = get_athena()
    result = athena.retrieve(query, mode=mode, top_k=top_k)
    return {**result.to_dict(), "relevant": athena.retriever.passes_relevance_gate(result)}


@mcp.tool(annotations={"readOnlyHint": True})
def ask(question: str) -> dict:
    """Answer a question from the runbooks with [n] citations, verification results, and a confidence score."""
    return get_athena().ask(question).to_dict()


if __name__ == "__main__":
    mcp.run()
