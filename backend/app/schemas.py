"""Request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- chat ------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    employee_id: str = "E-1001"


class Citation(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    document_title: str
    department: str
    source: str
    heading: str
    snippet: str
    score: float
    dense_score: float
    lexical_score: float


class TraceEntry(BaseModel):
    seq: int
    agent: str
    label: str
    status: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0


class ChatResponse(BaseModel):
    run_id: str
    conversation_id: str
    answer: str
    status: str
    intent: str
    confidence: float
    flags: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    used_citations: list[int] = Field(default_factory=list)
    proposed_action: dict[str, Any] | None = None
    approval_id: str | None = None
    requires_approval: bool = False
    verification: dict[str, Any] = Field(default_factory=dict)
    follow_up_question: str = ""
    trace: list[TraceEntry] = Field(default_factory=list)
    llm_mode: str = "demo"
    latency_ms: int = 0
    token_usage: dict[str, Any] | None = None


# --- conversations ---------------------------------------------------------
class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    run_id: str | None = None
    created_at: datetime


class ConversationOut(BaseModel):
    id: str
    title: str
    employee_id: str
    created_at: datetime
    message_count: int = 0


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)


# --- documents -------------------------------------------------------------
class DocumentOut(BaseModel):
    id: str
    title: str
    source: str
    doc_type: str
    department: str
    version: str
    effective_date: str | None = None
    chunk_count: int
    created_at: datetime


class IngestResponse(BaseModel):
    document_id: str
    title: str
    source: str
    chunks: int
    status: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: list[Citation]
    took_ms: int


# --- approvals -------------------------------------------------------------
class ApprovalOut(BaseModel):
    id: str
    run_id: str
    conversation_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk: str
    rationale: str
    flags: list[str] = Field(default_factory=list)
    confidence: float
    status: str
    requested_by: str
    decided_by: str | None = None
    decision_note: str = ""
    execution_result: dict[str, Any] | None = None
    created_at: datetime
    decided_at: datetime | None = None


class ApprovalDecision(BaseModel):
    decision: str = Field(description="approve | reject")
    decided_by: str = "manager@northwind.example"
    note: str = ""


# --- admin -----------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    llm_mode: str
    llm_model: str
    database: str
    vector_backend: str
    embedding_provider: str
    documents: int
    chunks: int


class MetricsResponse(BaseModel):
    runs_total: int
    runs_by_status: dict[str, int]
    approvals_by_status: dict[str, int]
    average_confidence: float
    average_latency_ms: float
    low_confidence_rate: float
    actions_executed: int
    top_documents: list[dict[str, Any]] = Field(default_factory=list)
    recent_runs: list[dict[str, Any]] = Field(default_factory=list)


class EmployeeOut(BaseModel):
    id: str
    name: str
    department: str
    manager: str
    location: str
    annual_remaining: float
    sick_remaining: float
