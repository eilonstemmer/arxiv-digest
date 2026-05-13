"""FastAPI application factory and startup configuration."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from .deps import DBDep, TemplatesDep, _get_db_path
from .routes import digests, papers, search, settings, trends, webhook

# Re-export so callers can `from src.main import DBDep, TemplatesDep` if desired.
# Routes should prefer `from src.deps import ...` to avoid circular imports.
__all__ = ["DBDep", "TemplatesDep", "app", "create_app"]

log = structlog.get_logger(__name__)


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    log.info("webapp.startup", db_path=str(_get_db_path()))
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="arxiv-digest",
        description="Local arXiv digest viewer",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=_lifespan,
    )

    application.include_router(digests.router)
    application.include_router(papers.router)
    application.include_router(settings.router)
    application.include_router(search.router)
    application.include_router(trends.router)
    application.include_router(webhook.router)

    return application


# Module-level app instance for uvicorn / docker CMD.
app = create_app()
