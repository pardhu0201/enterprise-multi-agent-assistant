"""Schema creation and seed data.

Runs on every boot and is idempotent, which is what lets a free container
(Hugging Face Space, Render) come up cold with a working corpus and demo
employees without a migration step.
"""

from __future__ import annotations

from sqlalchemy import select

from app.db.base import Base, engine, ensure_extensions, session_scope
from app.db.models import Employee
from app.logging_config import get_logger
from app.rag.ingest import ingest_corpus

log = get_logger(__name__)

SEED_EMPLOYEES = [
    {
        "id": "E-1001",
        "name": "Arjun Bathula",
        "email": "arjun.bathula@northwind.example",
        "department": "Engineering",
        "manager": "Priya Raman",
        "location": "Hyderabad",
        "annual_leave_total": 24.0,
        "annual_leave_used": 6.0,
        "sick_leave_total": 12.0,
        "sick_leave_used": 2.0,
    },
    {
        "id": "E-1002",
        "name": "Dana Whitfield",
        "email": "dana.whitfield@northwind.example",
        "department": "Finance",
        "manager": "Marcus Lee",
        "location": "London",
        "annual_leave_total": 24.0,
        "annual_leave_used": 19.0,
        "sick_leave_total": 12.0,
        "sick_leave_used": 0.0,
    },
    {
        "id": "E-1003",
        "name": "Sam Okonkwo",
        "email": "sam.okonkwo@northwind.example",
        "department": "Customer Success",
        "manager": "Priya Raman",
        "location": "Remote (Lagos)",
        "annual_leave_total": 24.0,
        "annual_leave_used": 1.0,
        "sick_leave_total": 12.0,
        "sick_leave_used": 5.0,
    },
]


def create_schema() -> None:
    ensure_extensions()
    Base.metadata.create_all(bind=engine)
    log.info("Database schema ready (%s)", engine.url.render_as_string(hide_password=True))


def seed_employees() -> int:
    created = 0
    with session_scope() as db:
        for record in SEED_EMPLOYEES:
            existing = db.execute(
                select(Employee).where(Employee.id == record["id"])
            ).scalar_one_or_none()
            if existing is None:
                db.add(Employee(**record))
                created += 1
    if created:
        log.info("Seeded %d employees", created)
    return created


def seed_corpus() -> int:
    with session_scope() as db:
        results = ingest_corpus(db)
    changed = sum(1 for r in results if r.status != "unchanged")
    log.info(
        "Corpus ready: %d documents, %d chunks (%d changed)",
        len(results),
        sum(r.chunks for r in results),
        changed,
    )
    return len(results)


def initialise(seed: bool = True) -> None:
    create_schema()
    if seed:
        seed_employees()
        seed_corpus()
