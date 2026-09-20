"""Health, configuration, graph topology and observability metrics."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import __version__
from app.agents.graph import graph_topology
from app.config import settings
from app.db.base import engine, get_session
from app.db.models import Approval, Chunk, Document, Employee, Run
from app.llm.client import get_llm
from app.schemas import EmployeeOut, HealthResponse, MetricsResponse
from app.tools.registry import tool_catalogue

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_session)):
    documents = db.execute(select(func.count(Document.id))).scalar() or 0
    chunks = db.execute(select(func.count(Chunk.id))).scalar() or 0
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_mode=get_llm().mode,
        llm_model=settings.anthropic_model if get_llm().available else "deterministic-fallback",
        database=engine.url.get_backend_name(),
        vector_backend=settings.vector_backend,
        embedding_provider=settings.embedding_provider,
        documents=documents,
        chunks=chunks,
    )


@router.get("/graph")
def graph():
    """Static agent topology, rendered by the UI's architecture panel."""
    return graph_topology()


@router.get("/tools")
def tools():
    return tool_catalogue()


@router.get("/employees", response_model=list[EmployeeOut])
def employees(db: Session = Depends(get_session)):
    rows = db.execute(select(Employee).order_by(Employee.id)).scalars()
    return [
        EmployeeOut(
            id=e.id,
            name=e.name,
            department=e.department,
            manager=e.manager,
            location=e.location,
            annual_remaining=e.annual_leave_total - e.annual_leave_used,
            sick_remaining=e.sick_leave_total - e.sick_leave_used,
        )
        for e in rows
    ]


@router.get("/metrics", response_model=MetricsResponse)
def metrics(db: Session = Depends(get_session)):
    runs = list(db.execute(select(Run).order_by(Run.created_at.desc()).limit(500)).scalars())
    approvals = list(db.execute(select(Approval)).scalars())

    confidences = [r.confidence for r in runs if r.confidence]
    latencies = [r.latency_ms for r in runs if r.latency_ms]
    low_confidence = [c for c in confidences if c < settings.confidence_threshold]

    document_hits: Counter = Counter()
    for run in runs:
        for citation in run.citations or []:
            document_hits[citation.get("document_title", "unknown")] += 1

    return MetricsResponse(
        runs_total=len(runs),
        runs_by_status=dict(Counter(r.status for r in runs)),
        approvals_by_status=dict(Counter(a.status for a in approvals)),
        average_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        average_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        low_confidence_rate=(
            round(len(low_confidence) / len(confidences), 3) if confidences else 0.0
        ),
        actions_executed=sum(1 for a in approvals if a.status == "approved"),
        top_documents=[
            {"title": title, "citations": count} for title, count in document_hits.most_common(6)
        ],
        recent_runs=[
            {
                "id": r.id,
                "query": r.query[:120],
                "intent": r.intent,
                "status": r.status,
                "confidence": r.confidence,
                "latency_ms": r.latency_ms,
                "llm_mode": r.llm_mode,
                "created_at": r.created_at,
            }
            for r in runs[:12]
        ],
    )
