"""Turn orchestration: run the graph, persist the trace, stream it to the UI.

`iter_turn` is the single execution path. The non-streaming API consumes the
same generator, so streamed and non-streamed responses can never drift apart.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.agents.graph import COMPILED_GRAPH
from app.agents.state import RunContext
from app.db.models import Conversation, Message, Run, RunEvent, utcnow
from app.llm.client import get_llm
from app.logging_config import get_logger

log = get_logger(__name__)

HISTORY_TURNS = 6


def _get_or_create_conversation(db: Session, conversation_id: str | None, employee_id: str):
    if conversation_id:
        conversation = db.get(Conversation, conversation_id)
        if conversation is not None:
            return conversation
    conversation = Conversation(employee_id=employee_id, title="New conversation")
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def _load_history(db: Session, conversation_id: str) -> list[dict[str, str]]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(desc(Message.created_at))
        .limit(HISTORY_TURNS)
    )
    rows = list(db.execute(stmt).scalars())
    return [{"role": m.role, "content": m.content} for m in reversed(rows)]


def _final_payload(state: dict[str, Any], run: Run) -> dict:
    return {
        "run_id": run.id,
        "conversation_id": run.conversation_id,
        "answer": state.get("answer", ""),
        "status": state.get("status", "completed"),
        "intent": state.get("intent", "question"),
        "confidence": state.get("confidence", 0.0),
        "flags": state.get("flags", []),
        "citations": [
            {k: v for k, v in item.items() if k != "content"}
            for item in (state.get("retrieved") or [])
        ],
        "used_citations": state.get("used_citations", []),
        "proposed_action": state.get("proposed_action"),
        "approval_id": state.get("approval_id"),
        "requires_approval": state.get("requires_approval", False),
        "verification": state.get("verification", {}),
        "follow_up_question": state.get("follow_up_question", ""),
        "trace": state.get("trace", []),
        "llm_mode": run.llm_mode,
        "latency_ms": run.latency_ms,
        "token_usage": run.token_usage,
    }


def iter_turn(
    db: Session,
    *,
    query: str,
    conversation_id: str | None = None,
    employee_id: str = "E-1001",
) -> Iterator[dict]:
    """Yield SSE-shaped events as the graph progresses, ending with `final`."""
    started = time.perf_counter()

    conversation = _get_or_create_conversation(db, conversation_id, employee_id)
    history = _load_history(db, conversation.id)

    llm = get_llm()
    run = Run(
        conversation_id=conversation.id,
        employee_id=employee_id,
        query=query,
        status="running",
        llm_mode=llm.mode,
    )
    db.add(run)
    db.add(Message(conversation_id=conversation.id, role="user", content=query, run_id=run.id))
    if conversation.title == "New conversation":
        conversation.title = query.strip()[:80]
    db.commit()
    db.refresh(run)

    yield {
        "event": "run_started",
        "data": {
            "run_id": run.id,
            "conversation_id": conversation.id,
            "llm_mode": llm.mode,
            "query": query,
        },
    }

    ctx = RunContext(db=db, llm=llm, employee_id=employee_id)
    config = {"configurable": {"ctx": ctx}, "recursion_limit": 25}
    initial = {
        "run_id": run.id,
        "conversation_id": conversation.id,
        "employee_id": employee_id,
        "query": query,
        "history": history,
        "trace": [],
    }

    state: dict[str, Any] = dict(initial)
    try:
        for update in COMPILED_GRAPH.stream(initial, config=config, stream_mode="updates"):
            for node_name, node_update in update.items():
                if not isinstance(node_update, dict):
                    continue
                events = node_update.get("trace") or []
                for event in events:
                    db.add(
                        RunEvent(
                            run_id=run.id,
                            seq=event["seq"],
                            agent=event["agent"],
                            status=event["status"],
                            summary=event["summary"],
                            payload=event["payload"],
                            duration_ms=event["duration_ms"],
                        )
                    )
                db.commit()

                merged_trace = state.get("trace", []) + events
                state.update({k: v for k, v in node_update.items() if k != "trace"})
                state["trace"] = merged_trace

                for event in events:
                    yield {"event": "agent_step", "data": {"node": node_name, **event}}
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Graph execution failed")
        run.status = "failed"
        run.answer = f"The assistant hit an internal error: {exc}"
        run.latency_ms = int((time.perf_counter() - started) * 1000)
        db.commit()
        yield {"event": "error", "data": {"run_id": run.id, "message": str(exc)}}
        return

    latency_ms = int((time.perf_counter() - started) * 1000)
    run.status = state.get("status", "completed")
    run.intent = state.get("intent", "question")
    run.answer = state.get("answer", "")
    run.citations = [
        {k: v for k, v in item.items() if k != "content"} for item in (state.get("retrieved") or [])
    ]
    run.proposed_action = state.get("proposed_action")
    run.verification = state.get("verification")
    run.confidence = float(state.get("confidence", 0.0))
    run.requires_approval = bool(state.get("requires_approval"))
    run.latency_ms = latency_ms
    run.token_usage = ctx.usage.as_dict()

    db.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content=run.answer,
            run_id=run.id,
        )
    )
    db.commit()
    db.refresh(run)

    yield {"event": "final", "data": _final_payload(state, run)}


def run_turn(
    db: Session,
    *,
    query: str,
    conversation_id: str | None = None,
    employee_id: str = "E-1001",
) -> dict:
    """Blocking variant - drives the same generator and returns the final payload."""
    payload: dict = {}
    for event in iter_turn(
        db, query=query, conversation_id=conversation_id, employee_id=employee_id
    ):
        if event["event"] in {"final", "error"}:
            payload = event["data"]
    return payload


def record_followup_message(db: Session, conversation_id: str, content: str, run_id: str) -> None:
    db.add(
        Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            run_id=run_id,
        )
    )
    db.commit()


def touch_run_status(db: Session, run_id: str, status: str) -> None:
    run = db.get(Run, run_id)
    if run is not None:
        run.status = status
        run.created_at = run.created_at or utcnow()
        db.commit()
