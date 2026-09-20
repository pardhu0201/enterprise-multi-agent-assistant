"""SQLAlchemy engine/session wiring.

Uses Postgres (+ pgvector) when `DATABASE_URL` is set and transparently falls
back to a local SQLite file otherwise, so the project runs with zero external
services during development and in CI.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings
from app.logging_config import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    pass


def _build_engine() -> Engine:
    url = settings.sqlalchemy_url
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 5
        kwargs["pool_recycle"] = 280  # Neon closes idle connections aggressively
    eng = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(eng, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - driver glue
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return eng


engine: Engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def ensure_extensions() -> None:
    """Create the pgvector extension when running on Postgres."""
    if not settings.is_postgres:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        log.info("pgvector extension is available")
    except Exception as exc:  # pragma: no cover - depends on DB privileges
        log.warning("Could not enable pgvector (%s); similarity falls back to numpy", exc)


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for non-request code (agents, scripts, tests)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
