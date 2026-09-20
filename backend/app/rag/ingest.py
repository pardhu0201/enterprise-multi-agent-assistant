"""Document ingestion: parse -> chunk -> embed -> persist.

Ingestion is idempotent: a document is keyed by its ``source`` and skipped when
the content checksum is unchanged, so restarting the server (or re-running the
seeder on every container boot) does not duplicate the corpus.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import CORPUS_DIR
from app.db.models import Chunk, Document
from app.logging_config import get_logger
from app.rag.chunking import chunk_document, parse_frontmatter
from app.rag.embeddings import get_embedder
from app.rag.retriever import invalidate_index

log = get_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class IngestResult:
    document_id: str
    title: str
    source: str
    chunks: int
    status: str  # created | updated | unchanged


def _checksum(text: str) -> str:
    """Content hash, salted with the embedding configuration.

    Re-ingestion is skipped when the checksum is unchanged. Hashing the text
    alone means switching embedder or dimension silently leaves stale vectors
    in the index, so the embedder's identity is part of the key.
    """
    embedder = get_embedder()
    fingerprint = f"{text}\x00{embedder.name}:{embedder.dim}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def read_pdf(path: Path) -> str:  # pragma: no cover - exercised manually
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def ingest_text(
    db: Session,
    *,
    raw: str,
    source: str,
    fallback_title: str,
    doc_type: str = "policy",
    department: str | None = None,
) -> IngestResult:
    parsed = parse_frontmatter(raw, fallback_title)
    checksum = _checksum(parsed.body)

    document = db.execute(select(Document).where(Document.source == source)).scalar_one_or_none()
    if document is not None and document.checksum == checksum:
        return IngestResult(document.id, document.title, source, document.chunk_count, "unchanged")

    status = "updated" if document is not None else "created"
    if document is None:
        document = Document(source=source)
        db.add(document)

    document.title = parsed.title
    document.doc_type = parsed.metadata.get("type", doc_type)
    document.department = department or parsed.metadata.get("department", "General")
    document.version = parsed.metadata.get("version", "1.0")
    document.effective_date = parsed.metadata.get("effective", None)
    document.checksum = checksum
    db.flush()

    db.execute(delete(Chunk).where(Chunk.document_id == document.id))

    pieces = chunk_document(parsed.title, parsed.body)
    embedder = get_embedder()
    vectors = embedder.embed_documents([p.content for p in pieces])

    for piece, vector in zip(pieces, vectors, strict=False):
        db.add(
            Chunk(
                document_id=document.id,
                ordinal=piece.ordinal,
                heading=piece.heading,
                content=piece.content,
                token_estimate=piece.token_estimate,
                embedding=[float(x) for x in vector],
            )
        )
    document.chunk_count = len(pieces)
    db.commit()
    invalidate_index()

    log.info("Ingested %-34s %-9s %3d chunks", source, status, len(pieces))
    return IngestResult(document.id, document.title, source, len(pieces), status)


def ingest_file(db: Session, path: Path, department: str | None = None) -> IngestResult:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")
    raw = read_pdf(path) if suffix == ".pdf" else path.read_text(encoding="utf-8")
    return ingest_text(
        db,
        raw=raw,
        source=path.name,
        fallback_title=path.stem.replace("-", " ").replace("_", " ").title(),
        department=department,
    )


def ingest_corpus(db: Session, corpus_dir: Path | None = None) -> list[IngestResult]:
    """Ingest every file in the bundled seed corpus."""
    directory = corpus_dir or CORPUS_DIR
    if not directory.exists():
        log.warning("Corpus directory %s does not exist", directory)
        return []
    results = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            results.append(ingest_file(db, path))
    return results


def delete_document(db: Session, document_id: str) -> bool:
    document = db.get(Document, document_id)
    if document is None:
        return False
    db.delete(document)
    db.commit()
    invalidate_index()
    return True
