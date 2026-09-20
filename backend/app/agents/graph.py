"""LangGraph wiring for the multi-agent assistant.

    START -> planner -> retrieval -> reasoning -+-> workflow -+-> verification
                            ^                   |             |
                            |                   +-------------+
                            |                                 |
                            +----- retry (max 1) -------------+
                                                              |
                                        approval_gate <-------+  (action proposed)
                                              |
                                             END

The graph is compiled once at import time and is stateless; per-request
services (DB session, LLM client, usage tracker) are injected through the
LangGraph config, so a single compiled graph serves every request safely.
"""

from __future__ import annotations

import time

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agents.planner import planner_node
from app.agents.reasoning_agent import reasoning_node
from app.agents.retrieval_agent import retrieval_node
from app.agents.state import AgentState, RunContext, trace_event
from app.agents.verification_agent import verification_node
from app.agents.workflow_agent import workflow_node
from app.db.models import Approval
from app.logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Human approval gate
# ---------------------------------------------------------------------------
def approval_gate_node(state: AgentState, config: RunnableConfig) -> dict:
    """Queue the prepared action for a human decision. Nothing is executed."""
    ctx: RunContext = config["configurable"]["ctx"]
    started = time.perf_counter()

    action = state.get("proposed_action") or {}
    verification = state.get("verification") or {}

    approval = Approval(
        run_id=state["run_id"],
        conversation_id=state.get("conversation_id", ""),
        tool_name=action.get("tool_name", "unknown"),
        arguments=action.get("arguments", {}),
        risk=action.get("risk", "medium"),
        rationale=action.get("rationale", ""),
        citations=state.get("used_citations", []),
        flags=state.get("flags", []),
        confidence=state.get("confidence", 0.0),
        status="pending",
        requested_by=ctx.employee_id,
    )
    ctx.db.add(approval)
    ctx.db.commit()
    ctx.db.refresh(approval)

    blockers = action.get("blockers") or []
    warnings = action.get("warnings") or []
    lines = [
        state.get("answer", "").strip(),
        "",
        f"**Action prepared - awaiting your approval:** `{action.get('tool_name')}`",
    ]
    if action.get("rationale"):
        lines.append(f"{action['rationale']}")
    if blockers:
        lines.append("")
        lines.append("**Blocked by policy:**")
        lines += [f"- {b}" for b in blockers]
    if warnings:
        lines.append("")
        lines.append("**Please confirm:**")
        lines += [f"- {w}" for w in warnings]
    lines.append("")
    lines.append("Nothing has been submitted yet. Approve or reject the request above.")

    return {
        "approval_id": approval.id,
        "answer": "\n".join(line for line in lines if line is not None).strip(),
        "status": "awaiting_approval",
        "requires_approval": True,
        "trace": [
            trace_event(
                ctx,
                "approval_gate",
                f"Queued '{approval.tool_name}' for human approval (risk={approval.risk})",
                status="warning",
                payload={
                    "approval_id": approval.id,
                    "tool_name": approval.tool_name,
                    "risk": approval.risk,
                    "blockers": blockers,
                    "warnings": warnings,
                    "confidence": verification.get("confidence"),
                },
                started=started,
            )
        ],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_after_reasoning(state: AgentState) -> str:
    if state.get("candidate_tool") and state.get("intent") in {"action", "mixed"}:
        return "workflow"
    return "verification"


def route_after_verification(state: AgentState) -> str:
    decision = (state.get("verification") or {}).get("decision", "answer")
    if decision == "retry":
        return "retrieval"
    if decision == "approval":
        return "approval_gate"
    return END


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("workflow", workflow_node)
    graph.add_node("verification", verification_node)
    graph.add_node("approval_gate", approval_gate_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "reasoning")
    graph.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {"workflow": "workflow", "verification": "verification"},
    )
    graph.add_edge("workflow", "verification")
    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {"retrieval": "retrieval", "approval_gate": "approval_gate", END: END},
    )
    graph.add_edge("approval_gate", END)

    return graph.compile()


COMPILED_GRAPH = build_graph()


def graph_topology() -> dict:
    """Static description of the graph, rendered by the UI."""
    return {
        "nodes": [
            {"id": "planner", "label": "Planner", "role": "Routes intent, writes queries"},
            {"id": "retrieval", "label": "Retrieval agent", "role": "Hybrid BM25 + vector search"},
            {"id": "reasoning", "label": "Reasoning agent", "role": "Grounded, cited synthesis"},
            {"id": "workflow", "label": "Workflow agent", "role": "Prepares validated action"},
            {
                "id": "verification",
                "label": "Verification agent",
                "role": "Groundedness + policy checks",
            },
            {"id": "approval_gate", "label": "Human approval", "role": "Blocks sensitive actions"},
        ],
        "edges": [
            {"source": "planner", "target": "retrieval"},
            {"source": "retrieval", "target": "reasoning"},
            {"source": "reasoning", "target": "workflow", "condition": "action intent"},
            {"source": "reasoning", "target": "verification", "condition": "question intent"},
            {"source": "workflow", "target": "verification"},
            {"source": "verification", "target": "retrieval", "condition": "evidence too thin"},
            {"source": "verification", "target": "approval_gate", "condition": "action proposed"},
        ],
    }
