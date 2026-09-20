"""Relational schema for documents, agent runs, approvals and the mock HRIS.

The ``embedding`` column is a real ``vector(N)`` when running on Postgres with
pgvector, and a JSON float array on SQLite. Everything else is portable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import settings
from app.db.base import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


# --- Embedding column type -------------------------------------------------
if settings.vector_backend == "pgvector":  # pragma: no cover - env dependent
    try:
        from pgvector.sqlalchemy import Vector

        EmbeddingColumn = Vector(settings.embedding_dim)
        EMBEDDING_IS_NATIVE = True
    except Exception:  # pgvector unavailable -> portable JSON storage
        EmbeddingColumn = JSON
        EMBEDDING_IS_NATIVE = False
else:
    EmbeddingColumn = JSON
    EMBEDDING_IS_NATIVE = False


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    source: Mapped[str] = mapped_column(String(500), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), default="policy")
    department: Mapped[str] = mapped_column(String(80), default="General")
    version: Mapped[str] = mapped_column(String(40), default="1.0")
    effective_date: Mapped[str | None] = mapped_column(String(40), nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("source", name="uq_document_source"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    heading: Mapped[str] = mapped_column(String(300), default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    embedding = mapped_column(EmbeddingColumn, nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (Index("ix_chunks_doc_ordinal", "document_id", "ordinal"),)


# ---------------------------------------------------------------------------
# Conversations and agent runs
# ---------------------------------------------------------------------------
class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    employee_id: Mapped[str] = mapped_column(String(40), default="E-1001", index=True)
    title: Mapped[str] = mapped_column(String(300), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))  # user | assistant | system
    content: Mapped[str] = mapped_column(Text, default="")
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Run(Base):
    """One pass of the multi-agent graph over a single user turn."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(String(32), index=True)
    employee_id: Mapped[str] = mapped_column(String(40), default="E-1001")
    query: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(40), default="question")
    status: Mapped[str] = mapped_column(String(40), default="running")
    answer: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list] = mapped_column(JSON, default=list)
    proposed_action: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    verification: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_mode: Mapped[str] = mapped_column(String(20), default="demo")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    token_usage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RunEvent(Base):
    """Per-agent trace entry - what the UI timeline renders."""

    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    agent: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), default="ok")
    summary: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------
class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    conversation_id: Mapped[str] = mapped_column(String(32), default="")
    tool_name: Mapped[str] = mapped_column(String(80))
    arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    risk: Mapped[str] = mapped_column(String(20), default="medium")
    rationale: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list] = mapped_column(JSON, default=list)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    requested_by: Mapped[str] = mapped_column(String(40), default="E-1001")
    decided_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    decision_note: Mapped[str] = mapped_column(Text, default="")
    execution_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    actor: Mapped[str] = mapped_column(String(80), default="system")
    action: Mapped[str] = mapped_column(String(80))
    entity: Mapped[str] = mapped_column(String(80), default="")
    entity_id: Mapped[str] = mapped_column(String(64), default="")
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# Mock enterprise systems the workflow agent acts on
# ---------------------------------------------------------------------------
class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(160), default="")
    department: Mapped[str] = mapped_column(String(80), default="Engineering")
    manager: Mapped[str] = mapped_column(String(120), default="Priya Raman")
    location: Mapped[str] = mapped_column(String(80), default="Bengaluru")
    annual_leave_total: Mapped[float] = mapped_column(Float, default=24.0)
    annual_leave_used: Mapped[float] = mapped_column(Float, default=0.0)
    sick_leave_total: Mapped[float] = mapped_column(Float, default=12.0)
    sick_leave_used: Mapped[float] = mapped_column(Float, default=0.0)


class LeaveRequest(Base):
    __tablename__ = "leave_requests"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    employee_id: Mapped[str] = mapped_column(String(40), index=True)
    leave_type: Mapped[str] = mapped_column(String(40), default="annual")
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    working_days: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="submitted")
    approval_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExpenseClaim(Base):
    __tablename__ = "expense_claims"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    employee_id: Mapped[str] = mapped_column(String(40), index=True)
    category: Mapped[str] = mapped_column(String(40), default="travel")
    amount: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(8), default="INR")
    description: Mapped[str] = mapped_column(Text, default="")
    incurred_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="submitted")
    approval_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ITTicket(Base):
    __tablename__ = "it_tickets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    employee_id: Mapped[str] = mapped_column(String(40), index=True)
    category: Mapped[str] = mapped_column(String(60), default="access")
    priority: Mapped[str] = mapped_column(String(20), default="normal")
    summary: Mapped[str] = mapped_column(String(300), default="")
    details: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="open")
    approval_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
