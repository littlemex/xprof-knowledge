"""xprof-knowledge — a FastMCP server that serves cross-accelerator profiling & tuning know-how.

One knowledge surface for GPU, Neuron, and cross-cutting methodology, so an agent driving the
profile -> analyze -> improve -> re-experiment loop retrieves the relevant playbook over MCP. It is
self-contained: pure Python, no accelerator, no cloud, no orchestrator — install it and run
`xprof-knowledge-mcp`, then connect any MCP client over streamable-http.
"""
from __future__ import annotations

import os
from typing import Any

from .library import CHIPS, KnowledgeLibrary, default_root

SERVER_NAME = "xprof-knowledge"
DEFAULT_HOST = "0.0.0.0"


def build_server(library: KnowledgeLibrary, port: int = 8080) -> Any:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(SERVER_NAME, host=DEFAULT_HOST, port=port)

    def _check_chip(chip: str) -> None:
        if chip and chip not in CHIPS:
            raise ValueError(f"unknown chip {chip!r}; use one of {sorted(CHIPS)} or omit it")

    @mcp.tool()
    def list_topics(chip: str = "") -> dict:
        """List the available playbooks, optionally filtered by chip ('gpu' | 'neuron' | 'common').
        Returns each topic's id, chip, title, and tags — call get_topic(id) for the full text."""
        _check_chip(chip)
        ts = library.list_topics(chip or None)
        return {"chip": chip or "all", "count": len(ts), "topics": [
            {"id": t.id, "chip": t.chip, "title": t.title, "tags": list(t.tags)} for t in ts]}

    @mcp.tool()
    def get_topic(topic_id: str) -> dict:
        """Return one playbook in full (title, body, reference links). topic_id is like
        'gpu/roofline' — get it from list_topics or search_knowledge."""
        t = library.get(topic_id)
        return {"id": t.id, "chip": t.chip, "title": t.title, "tags": list(t.tags),
                "refs": list(t.refs), "body": t.body}

    @mcp.tool()
    def search_knowledge(query: str, chip: str = "", limit: int = 5) -> dict:
        """Find the playbooks most relevant to a symptom or question (keyword ranked), optionally
        scoped to a chip. Returns ranked id/title/snippet — then get_topic(id) for the full text.
        Example: search_knowledge('kernel is memory bound and occupancy is high', chip='gpu').
        The playbooks are in English; search with English keywords."""
        _check_chip(chip)
        hits = library.search(query, chip or None, limit=limit)
        out = {"query": query, "chip": chip or "all", "count": len(hits), "results": [
            {"id": t.id, "chip": t.chip, "title": t.title, "score": s, "snippet": t.snippet()}
            for t, s in hits]}
        if not hits:
            out["hint"] = ("no match — the playbooks are English; try English keywords "
                           "(e.g. roofline, coalescing, fusion, occupancy, dma, collective, mfu), "
                           "or list_topics() to browse.")
        return out

    @mcp.resource("knowledge://index")
    def index_resource() -> str:
        """A browsable index of every playbook (template resources are not enumerable on their own)."""
        import json
        return json.dumps([{"id": t.id, "chip": t.chip, "title": t.title, "tags": list(t.tags)}
                           for t in library.list_topics()], indent=2)

    @mcp.resource("knowledge://{chip}/{slug}")
    def topic_resource(chip: str, slug: str) -> str:
        """Expose each playbook as an MCP resource so resource-aware clients can browse/read it."""
        return library.get(f"{chip}/{slug}").body

    @mcp.prompt()
    def diagnose(chip: str, symptom: str) -> str:
        """A prompt that primes an agent to diagnose a profiling symptom using this knowledge base."""
        return (f"You are diagnosing a {chip} performance profile. Symptom: {symptom}\n"
                f"1. Call search_knowledge with the symptom (chip='{chip}').\n"
                f"2. Read the top playbook(s) with get_topic.\n"
                f"3. State the most likely cause, the exact metric/section to confirm it in the "
                f"profile, and the single change to try next — then re-run and compare.")

    return mcp


def main() -> None:
    port = int(os.environ.get("MCP_PORT", "8080"))
    library = KnowledgeLibrary(default_root())
    build_server(library, port=port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
