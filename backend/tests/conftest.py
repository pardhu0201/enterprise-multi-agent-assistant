"""Test fixtures.

The application builds its engine and settings at import time, so the test
database must be configured through the environment *before* any `app` module
is imported. Everything below therefore imports lazily inside fixtures.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import pytest

TEST_DB = Path(tempfile.gettempdir()) / "northwind_assistant_test.db"

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"
os.environ["ANTHROPIC_API_KEY"] = ""  # force deterministic demo mode in CI
os.environ["SEED_ON_STARTUP"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["EMBEDDING_PROVIDER"] = "hashing"


@pytest.fixture(scope="session", autouse=True)
def _database():
    if TEST_DB.exists():
        TEST_DB.unlink()
    from app.db.init_db import initialise

    initialise(seed=True)
    yield
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(TEST_DB) + suffix)
        if candidate.exists():
            # Windows may still hold a handle until the connection is GC'd.
            with contextlib.suppress(PermissionError):
                candidate.unlink()


@pytest.fixture
def db():
    from app.db.base import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
