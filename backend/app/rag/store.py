"""Vector storage and dense candidate lookup.

On Postgres the ``chunks.embedding`` column is a native ``vector(N)`` and the
nearest-neighbour search is pushed down to pgvector's cosine operator. On
SQLite (the zero-setup default) embeddings are JSON arrays and the same search
runs in-process with numpy. The retriever above this layer does not care which
one answered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import EMBEDDING_IS_NATIVE, Chunk, Document
from app.logging_config import get_logger
from app.rag.embeddings import cosine_matrix

log = get_logger(__name__)


@dataclass
class ScoredChunk:
    chunk_id: str
    document_id: str
    document_title: str
    department: str
    source: str
    heading: str
    content: str
    score: float = 0.0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rank: int = 0

    def to_citation(self, index: int) -> dict:
        return {
            "index": index,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "department": self.department,
            "source": self.source,
            "heading": self.heading,
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "dense_score": round(self.dense_score, 4),
            "lexical_score": round(self.lexical_score, 4),
        }

    @property
    def snippet(self) -> str:
        body = self.content.split("]\n", 1)[-1] if self.content.startswith("[") else self.content
        body = " ".join(body.split())
        return body[:420] + ("..." if len(body) > 420 else "")


def _row_to_scored(chunk: Chunk, document: Document) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk.id,
        document_id=document.id,
        document_title=document.title,
        department=document.department,
        source=document.source,
        heading=chunk.heading,
        content=chunk.content,
    )


def load_all_chunks(db: Session) -> list[tuple[Chunk, Document]]:
    stmt = select(Chunk, Document).join(Document, Chunk.document_id == Document.id)
    return list(db.execute(stmt).all())


def dense_search(db: Session, query_vector: np.ndarray, limit: int) -> list[ScoredChunk]:
    """Top-``limit`` chunks by cosine similarity."""
    if EMBEDDING_IS_NATIVE:  # pragma: no cover - requires a live Postgres
        try:
            distance = Chunk.embedding.cosine_distance(query_vector.tolist())
            stmt = (
                select(Chunk, Document, distance.label("distance"))
                .join(Document, Chunk.document_id == Document.id)
                .order_by(distance)
                .limit(limit)
            )
            results: list[ScoredChunk] = []
            for chunk, document, dist in db.execute(stmt).all():
                scored = _row_to_scored(chunk, document)
                scored.dense_score = float(1.0 - (dist or 0.0))
                results.append(scored)
            return results
        except Exception as exc:
            log.warning("pgvector search failed (%s); using in-process cosine", exc)

    rows = load_all_chunks(db)
    if not rows:
        return []
    vectors = np.asarray(
        [(r[0].embedding if r[0].embedding is not None else []) for r in rows],
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        return []
    sims = cosine_matrix(query_vector.astype(np.float32), vectors)
    order = np.argsort(-sims)[:limit]
    results = []
    for i in order:
        chunk, document = rows[int(i)]
        scored = _row_to_scored(chunk, document)
        scored.dense_score = float(sims[int(i)])
        results.append(scored)
    return results


def corpus_snapshot(db: Session) -> list[ScoredChunk]:
    """Every chunk, unscored - used to build the in-memory BM25 index."""
    return [_row_to_scored(chunk, document) for chunk, document in load_all_chunks(db)]


def corpus_version(db: Session) -> str:
    """Cheap cache key that changes whenever the corpus changes."""
    from sqlalchemy import func

    count = db.execute(select(func.count(Chunk.id))).scalar() or 0
    latest = db.execute(select(func.max(Document.created_at))).scalar()
    return f"{count}:{latest}"
