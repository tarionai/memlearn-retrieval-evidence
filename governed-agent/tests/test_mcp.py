"""Phase 2 acceptance — the MCP tool surface, exercised through an in-process
MCP client session (the real protocol round-trip, not a direct function call)."""
from __future__ import annotations

import asyncio

from mcp.shared.memory import create_connected_server_and_client_session as connected_session

from governed_agent import chain
from governed_agent.mcp_server import mcp


def _forbidden_claim() -> str:
    boundaries = chain.load_frozen_report()["claim_boundaries"]
    return next(b["claim"] for b in boundaries if b["status"] == "Not validated")


def _run(coro):
    return asyncio.run(coro)


def test_check_claim_citable_blocks_not_validated_claim():
    async def scenario():
        async with connected_session(mcp) as client:
            await client.initialize()
            tools = {t.name for t in (await client.list_tools()).tools}
            assert {"run_governed_retrieval", "check_claim_citable", "read_audit_log"} <= tools

            result = await client.call_tool("check_claim_citable", {"claim": _forbidden_claim()})
            return result.structuredContent

    payload = _run(scenario())
    assert payload["citable"] is False
    assert "not citable" in payload["reason"]
    # The rule that disposed of the claim is named in the contract.
    assert payload["rule"] == chain.CITABILITY_RULE


def test_check_claim_citable_allows_unforbidden_claim():
    async def scenario():
        async with connected_session(mcp) as client:
            await client.initialize()
            result = await client.call_tool(
                "check_claim_citable", {"claim": "retrieval over a Postgres full-text index"}
            )
            return result.structuredContent

    payload = _run(scenario())
    assert payload["citable"] is True
