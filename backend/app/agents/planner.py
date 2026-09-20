"""Planner agent - routes the turn and writes the retrieval queries."""

from __future__ import annotations

import time

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.prompts import PLANNER_SYSTEM
from app.agents.state import AgentState, RunContext, trace_event
from app.logging_config import get_logger
from app.rag.embeddings import tokenize
from app.tools.registry import TOOLS

log = get_logger(__name__)

QUESTION_MARKERS = (
    "what",
    "how",
    "when",
    "where",
    "why",
    "who",
    "can i",
    "am i",
    "do i",
    "is there",
    "explain",
    "summarise",
    "summarize",
    "tell me",
    "policy",
)
# Phrases that signal the employee is asking for something to be *done*, as
# opposed to asking about the rule. "How long do I have to file a claim?"
# mentions filing but is a question, so a bare verb is not enough.
REQUEST_MARKERS = (
    "please",
    "can you",
    "could you",
    "help me",
    "i want to",
    "i need to",
    "i would like to",
    "i'd like to",
    "for me",
    "book me",
    "put in",
    "go ahead",
    "on my behalf",
    "let's submit",
    "lets submit",
)
# ...unless the sentence opens with one of these, which makes it an imperative.
IMPERATIVE_VERBS = (
    "submit",
    "book",
    "apply",
    "raise",
    "open",
    "file",
    "create",
    "log",
    "request",
    "claim",
    "take",
    "cancel",
    "start",
)


class PlannerOutput(BaseModel):
    intent: str = Field(description="question | action | mixed | smalltalk")
    rationale: str = Field(description="One sentence explaining the routing decision.")
    search_queries: list[str] = Field(
        default_factory=list, description="1-3 keyword-rich retrieval queries."
    )
    candidate_tool: str = Field(default="", description="Best matching tool name, or empty string.")


def _tool_catalogue_text() -> str:
    return "\n".join(
        f"- {spec.name} (risk={spec.risk}): {spec.description}" for spec in TOOLS.values()
    )


def rule_based_plan(query: str) -> PlannerOutput:
    """Deterministic planner used in demo mode and as the API fallback."""
    lowered = query.lower().strip()
    tokens = set(tokenize(lowered))

    if len(tokens) <= 3 and any(
        g in lowered for g in ("hi", "hello", "hey", "thanks", "thank you", "good morning")
    ):
        return PlannerOutput(
            intent="smalltalk",
            rationale="Greeting with no policy question or action.",
            search_queries=[],
            candidate_tool="",
        )

    best_tool, best_hits = "", 0
    for spec in TOOLS.values():
        hits = sum(1 for kw in spec.keywords if kw in lowered)
        if hits > best_hits:
            best_tool, best_hits = spec.name, hits

    first_word = (tokenize(lowered) or [""])[0]
    wants_action = any(marker in lowered for marker in REQUEST_MARKERS) or (
        first_word in IMPERATIVE_VERBS
    )
    asks_question = any(marker in lowered for marker in QUESTION_MARKERS) or "?" in query

    if wants_action and best_tool:
        intent = "mixed" if asks_question else "action"
    elif asks_question:
        intent = "question"
    elif best_tool and best_hits >= 2:
        intent = "action"
    else:
        intent = "question"

    candidate_tool = best_tool if intent in {"action", "mixed"} else ""

    queries = [query.strip()]
    if best_tool:
        spec = TOOLS[best_tool]
        queries.append(" ".join(spec.keywords[:4]) + " policy")

    return PlannerOutput(
        intent=intent,
        rationale=(
            f"Keyword routing: action markers={wants_action}, question markers={asks_question}"
            + (f", tool match='{best_tool}'" if best_tool else "")
        ),
        search_queries=queries[:3],
        candidate_tool=candidate_tool,
    )


def planner_node(state: AgentState, config: RunnableConfig) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    started = time.perf_counter()
    query = state["query"]

    history = state.get("history") or []
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:]) or "(none)"

    user = (
        f"Tool catalogue:\n{_tool_catalogue_text()}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Employee request:\n{query}"
    )

    result = ctx.llm.structured(
        agent="planner",
        system=PLANNER_SYSTEM,
        user=user,
        schema=PlannerOutput,
        fallback=lambda: rule_based_plan(query),
        max_tokens=1500,
    )
    ctx.usage.add("planner", result)
    plan: PlannerOutput = result.value  # type: ignore[assignment]

    # The model may name a tool that does not exist; the registry is authoritative.
    candidate_tool = plan.candidate_tool if plan.candidate_tool in TOOLS else ""
    queries = [q for q in plan.search_queries if q.strip()] or [query]

    return {
        "intent": plan.intent,
        "plan_rationale": plan.rationale,
        "search_queries": queries[:3],
        "candidate_tool": candidate_tool,
        "retrieval_attempt": 0,
        "trace": [
            trace_event(
                ctx,
                "planner",
                f"Intent '{plan.intent}'"
                + (f", tool '{candidate_tool}'" if candidate_tool else ", no tool required"),
                payload={
                    "intent": plan.intent,
                    "rationale": plan.rationale,
                    "search_queries": queries[:3],
                    "candidate_tool": candidate_tool,
                    "mode": result.mode,
                },
                started=started,
            )
        ],
    }
