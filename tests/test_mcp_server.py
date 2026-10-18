"""Checks for mcp_server.py: the tools it exposes and how they are marked.

Skipped when the optional `mcp` package is not installed (requirements-mcp.txt).
"""
import asyncio

import pytest

pytest.importorskip("mcp")
import mcp_server  # noqa: E402  (needs the importorskip above)
import tools  # noqa: E402


def test_exposes_every_read_tool_and_both_actions():
    listed = {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert set(listed) == set(tools.TOOLS) | {"open_incident", "rollback"}
    # Clients ask before running a write; read tools can run freely.
    for name, t in listed.items():
        assert t.annotations.read_only_hint == (name in tools.TOOLS), name
