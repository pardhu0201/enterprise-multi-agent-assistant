"""FastAPI application entrypoint.

Serves the JSON API under `/api` and, when a built frontend is present, the
React single-page app from the same origin - which is what lets the whole
project run as one free container.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import routes_admin, routes_approvals, routes_chat, routes_documents
from app.config import BACKEND_DIR, settings
from app.db.init_db import initialise
from app.logging_config import configure_logging, get_logger

configure_logging(settings.log_level)
log = get_logger(__name__)

FRONTEND_DIST = BACKEND_DIR / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info(
        "Starting Enterprise Multi-Agent Assistant v%s (env=%s, llm=%s, vectors=%s)",
        __version__,
        settings.app_env,
        settings.llm_mode,
        settings.vector_backend,
    )
    initialise(seed=settings.seed_on_startup)
    yield
    log.info("Shutting down")


app = FastAPI(
    title="Enterprise Multi-Agent AI Assistant",
    description=(
        "Multi-agent RAG assistant for enterprise policy Q&A and workflow automation. "
        "Retrieval, reasoning, workflow and verification agents coordinated with "
        "LangGraph, with a mandatory human approval gate before any business action."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_chat.router, prefix="/api")
app.include_router(routes_documents.router, prefix="/api")
app.include_router(routes_approvals.router, prefix="/api")
app.include_router(routes_admin.router, prefix="/api")


@app.get("/api", include_in_schema=False)
def api_root():
    return JSONResponse(
        {
            "name": "Enterprise Multi-Agent AI Assistant",
            "version": __version__,
            "docs": "/api/docs",
            "llm_mode": settings.llm_mode,
        }
    )


# --- static frontend (single-container deployment) -------------------------
def _mount_frontend(directory: Path) -> None:
    assets = directory / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = directory / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        candidate = directory / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


if settings.serve_frontend and (FRONTEND_DIST / "index.html").is_file():
    log.info("Serving built frontend from %s", FRONTEND_DIST)
    _mount_frontend(FRONTEND_DIST)
else:

    @app.get("/", include_in_schema=False)
    def root():
        return JSONResponse(
            {
                "name": "Enterprise Multi-Agent AI Assistant",
                "version": __version__,
                "message": "API is running. The React UI is served separately in dev mode.",
                "docs": "/api/docs",
                "health": "/api/health",
            }
        )
