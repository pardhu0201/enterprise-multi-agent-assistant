"""Chat endpoints - blocking and server-sent-events streaming."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.runner import iter_turn, run_turn
from app.db.base import SessionLocal, get_session
from app.db.models import Conversation, Message, Run
from app.logging_config import get_logger
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationOut,
    MessageOut,
)

log = get_logger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_session)) -> ChatResponse:
    """Run the full agent graph and return the verified result."""
    result = run_turn(
        db,
        query=payload.message,
        conversation_id=payload.conversation_id,
        employee_id=payload.employee_id,
    )
    if not result:
        raise HTTPException(status_code=500, detail="The assistant produced no result")
    return ChatResponse(**result)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """Stream each agent's step as it completes.

    The generator owns its own session: a streaming response outlives the
    request-scoped dependency, so reusing that session would close it mid-run.
    """

    def event_source() -> Iterator[str]:
        db = SessionLocal()
        try:
            for event in iter_turn(
                db,
                query=payload.message,
                conversation_id=payload.conversation_id,
                employee_id=payload.employee_id,
            ):
                yield _sse(event["event"], event["data"])
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Streaming turn failed")
            yield _sse("error", {"message": str(exc)})
        finally:
            db.close()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    employee_id: str | None = None, limit: int = 30, db: Session = Depends(get_session)
):
    counts = dict(
        db.execute(
            select(Message.conversation_id, func.count(Message.id)).group_by(
                Message.conversation_id
            )
        ).all()
    )
    stmt = select(Conversation).order_by(Conversation.created_at.desc()).limit(limit)
    if employee_id:
        stmt = stmt.where(Conversation.employee_id == employee_id)
    return [
        ConversationOut(
            id=c.id,
            title=c.title,
            employee_id=c.employee_id,
            created_at=c.created_at,
            message_count=counts.get(c.id, 0),
        )
        for c in db.execute(stmt).scalars()
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, db: Session = Depends(get_session)):
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    ).scalars()
    items = [
        MessageOut(
            id=m.id, role=m.role, content=m.content, run_id=m.run_id, created_at=m.created_at
        )
        for m in messages
    ]
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        employee_id=conversation.employee_id,
        created_at=conversation.created_at,
        message_count=len(items),
        messages=items,
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_session)):
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conversation)
    db.commit()
    return {"deleted": conversation_id}


@router.get("/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_session)):
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    from app.db.models import RunEvent

    events = db.execute(
        select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
    ).scalars()
    return {
        "id": run.id,
        "conversation_id": run.conversation_id,
        "query": run.query,
        "intent": run.intent,
        "status": run.status,
        "answer": run.answer,
        "confidence": run.confidence,
        "citations": run.citations,
        "proposed_action": run.proposed_action,
        "verification": run.verification,
        "requires_approval": run.requires_approval,
        "llm_mode": run.llm_mode,
        "latency_ms": run.latency_ms,
        "token_usage": run.token_usage,
        "created_at": run.created_at,
        "trace": [
            {
                "seq": e.seq,
                "agent": e.agent,
                "status": e.status,
                "summary": e.summary,
                "payload": e.payload,
                "duration_ms": e.duration_ms,
            }
            for e in events
        ],
    }
