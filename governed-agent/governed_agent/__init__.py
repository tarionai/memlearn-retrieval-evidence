"""Governed retrieval agent — LangGraph state graph + MCP server over the
cold-verified enterprise-adapter chain.

The chain logic (ingest -> retrieve -> evaluate), the deterministic gate, and
the append-only audit are NOT reimplemented here. They are imported verbatim
from the published, cold-reproducible enterprise-adapter package (see chain.py).
This package adds only the graph wiring, the optional LLM draft node, and the
MCP tool surface.
"""
