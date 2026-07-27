"""FastAPI application factory and `glassfit` console entrypoint."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from glassfit import __version__
from glassfit.api import feedback, frames, health, measurements, recommendations, scan
from glassfit.catalog.store import ensure_seeded
from glassfit.config import Settings, get_settings
from glassfit.errors import register_error_handlers
from glassfit.storage import db
from glassfit.storage.repo import Repo


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conn = db.connect(app_settings.db_path)
        repo = Repo(conn)
        ensure_seeded(repo, app_settings.seed_frames_path)
        app.state.settings = app_settings
        app.state.repo = repo
        app.state.detector = None  # built lazily by api.deps.get_detector
        app.state.detector_lock = threading.Lock()
        yield
        conn.close()

    app = FastAPI(title="GlassFit", version=__version__, lifespan=lifespan)
    register_error_handlers(app)
    for router_module in (health, scan, measurements, recommendations, frames, feedback):
        app.include_router(router_module.router, prefix="/api/v1")
    # Mounted LAST so /api/v1/* always wins over the static catch-all.
    if app_settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=app_settings.frontend_dir, html=True))
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("glassfit.app:app", host=settings.host, port=settings.port, reload=settings.dev)
