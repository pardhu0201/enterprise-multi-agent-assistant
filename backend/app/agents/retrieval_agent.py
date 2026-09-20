"""Retrieval agent - runs hybrid search and assembles the evidence block.

Retrieval is deterministic and identical in Claude and demo mode: the planner
supplies the queries, this node fuses BM25 and vector results, deduplicates
across queries, and packs the highest-scoring chunks into a token budget.

On a re-entry (the verifier can send the run back once) it broadens the search
instead of repeating it - more candidates, more queries, looser diversity.
"""

from __future__ import annotations

import time

from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, RunContext, trace_event
from app.config import settings
from app.logging_config import get_logger
from app.rag.retriever import build_context_block, retrieve
from app.rag.store import ScoredChunk

log = get_logger(__name__)


def _merge(results: list[list[ScoredChunk]], limit: int) -> list[ScoredChunk]:
    """Interleave per-query result lists, keeping the best score per chunk."""
    merged: dict[str, ScoredChunk] = {}
    for result in results:
        for chunk in result:
            existing = merged.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                merged[chunk.chunk_id] = chunk
    ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)[:limit]
    for position, chunk in enumerate(ranked, start=1):
        chunk.rank = position
    return ranked


def retrieval_node(state: AgentState, config: RunnableConfig) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    started = time.perf_counter()

    attempt = state.get("retrieval_attempt", 0)
    queries = list(state.get("search_queries") or [state["query"]])

    if state.get("intent") == "smalltalk":
        return {
            "retrieved": [],
            "context_block": "",
            "retrieval_attempt": attempt + 1,
            "retrieval_note": "Skipped - conversational turn with no policy content.",
            "trace": [
                trace_event(
                    ctx,
                    "retrieval",
                    "Skipped retrieval (no policy content in request)",
                    payload={"queries": []},
                    started=started,
                )
            ],
        }

    widen = attempt > 0
    if widen:
        # Second pass: fall back to the raw question too, and cast a wider net.
        queries = list(dict.fromkeys([*queries, state["query"]]))
    top_k = settings.retrieval_top_k + (3 if widen else 0)
    candidates = settings.retrieval_candidates * (2 if widen else 1)

    per_query = [retrieve(ctx.db, q, top_k=top_k, candidates=candidates) for q in queries]
    chunks = _merge(per_query, top_k)
    context_block = build_context_block(chunks)

    citations = [chunk.to_citation(i) for i, chunk in enumerate(chunks, start=1)]
    retrieved = [
        {**citation, "content": chunk.content}
        for citation, chunk in zip(citations, chunks, strict=True)
    ]

    top_score = chunks[0].score if chunks else 0.0
    if chunks:
        sources = len({c.document_title for c in chunks})
        note = (
            f"{len(chunks)} passage{'s' if len(chunks) != 1 else ''} from "
            f"{sources} document{'s' if sources != 1 else ''}"
        )
    else:
        note = "No matching passages in the knowledge base"

    return {
        "retrieved": retrieved,
        "context_block": context_block,
        "retrieval_attempt": attempt + 1,
        "retrieval_note": note,
        "trace": [
            trace_event(
                ctx,
                "retrieval",
                ("Broadened search: " if widen else "") + note,
                status="ok" if chunks else "warning",
                payload={
                    "queries": queries,
                    "top_k": top_k,
                    "top_score": round(top_score, 3),
                    "documents": sorted({c.document_title for c in chunks}),
                    "citations": citations,
                },
                started=started,
            )
        ],
    }
