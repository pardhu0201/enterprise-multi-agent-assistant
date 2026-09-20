"""Shared state passed between agents in the LangGraph workflow.

`trace` uses an additive reducer so every node can append its own timeline
entry without clobbering earlier ones - that list is exactly what the UI
renders and what gets persisted to `run_events`.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from sqlalchemy.orm import Session

from app.llm.client import LLMClient, UsageTracker


class AgentState(TypedDict, total=False):
    # --- inputs ---
    run_id: str
    conversation_id: str
    employee_id: str
    query: str
    history: list[dict[str, str]]

    # --- planner ---
    intent: str
    plan_rationale: str
    search_queries: list[str]
    candidate_tool: str

    # --- retrieval ---
    retrieved: list[dict[str, Any]]
    context_block: str
    retrieval_attempt: int
    retrieval_note: str

    # --- reasoning ---
    answer: str
    used_citations: list[int]
    insufficient_evidence: bool
    follow_up_question: str

    # --- workflow ---
    proposed_action: dict[str, Any] | None
    action_note: str

    # --- verification ---
    verification: dict[str, Any]
    confidence: float
    flags: list[str]
    requires_approval: bool
    approval_id: str
    status: str

    # --- bookkeeping ---
    trace: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class RunContext:
    """Per-request services handed to nodes through the LangGraph config."""

    db: Session
    llm: LLMClient
    employee_id: str
    usage: UsageTracker = field(default_factory=UsageTracker)
    _seq: int = 0

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq


AGENT_LABELS = {
    "planner": "Planner",
    "retrieval": "Retrieval agent",
    "reasoning": "Reasoning agent",
    "workflow": "Workflow agent",
    "verification": "Verification agent",
    "approval_gate": "Human approval gate",
}


def trace_event(
    ctx: RunContext,
    agent: str,
    summary: str,
    *,
    status: str = "ok",
    payload: dict | None = None,
    started: float | None = None,
) -> dict[str, Any]:
    return {
        "seq": ctx.next_seq(),
        "agent": agent,
        "label": AGENT_LABELS.get(agent, agent.title()),
        "status": status,
        "summary": summary,
        "payload": payload or {},
        "duration_ms": int((time.perf_counter() - started) * 1000) if started else 0,
    }
