"""Knowledge-base endpoints: list, upload, reindex, delete, and search preview."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import get_session
from app.db.models import Document
from app.logging_config import get_logger
from app.rag.ingest import SUPPORTED_SUFFIXES, delete_document, ingest_corpus, ingest_text
from app.rag.retriever import retrieve
from app.schemas import Citation, DocumentOut, IngestResponse, SearchRequest, SearchResponse

log = get_logger(__name__)
router = APIRouter(tags=["knowledge-base"])

MAX_UPLOAD_BYTES = 2 * 1024 * 1024


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_session)):
    documents = db.execute(select(Document).order_by(Document.title)).scalars()
    return [
        DocumentOut(
            id=d.id,
            title=d.title,
            source=d.source,
            doc_type=d.doc_type,
            department=d.department,
            version=d.version,
            effective_date=d.effective_date,
            chunk_count=d.chunk_count,
            created_at=d.created_at,
        )
        for d in documents
    ]


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form("General"),
    db: Session = Depends(get_session),
):
    name = file.filename or "upload.md"
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
        )

    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 2 MB")

    if suffix == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(payload))
        raw = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        raw = payload.decode("utf-8", errors="replace")

    if not raw.strip():
        raise HTTPException(status_code=422, detail="No extractable text in that file")

    result = ingest_text(
        db,
        raw=raw,
        source=name,
        fallback_title=name.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title(),
        department=department,
    )
    return IngestResponse(
        document_id=result.document_id,
        title=result.title,
        source=result.source,
        chunks=result.chunks,
        status=result.status,
    )


@router.post("/documents/reindex", response_model=list[IngestResponse])
def reindex_corpus(db: Session = Depends(get_session)):
    """Re-ingest the bundled seed corpus (useful after changing the embedder)."""
    results = ingest_corpus(db)
    return [
        IngestResponse(
            document_id=r.document_id,
            title=r.title,
            source=r.source,
            chunks=r.chunks,
            status=r.status,
        )
        for r in results
    ]


@router.delete("/documents/{document_id}")
def remove_document(document_id: str, db: Session = Depends(get_session)):
    if not delete_document(db, document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": document_id}


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_session)):
    """Retrieval-only preview - shows what the retrieval agent would see."""
    started = time.perf_counter()
    chunks = retrieve(db, payload.query, top_k=payload.top_k)
    results = [Citation(**chunk.to_citation(i)) for i, chunk in enumerate(chunks, start=1)]
    return SearchResponse(
        query=payload.query,
        results=results,
        took_ms=int((time.perf_counter() - started) * 1000),
    )
