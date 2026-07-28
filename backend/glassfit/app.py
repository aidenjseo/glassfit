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
        app.state.detector = None  # built by the prewarm below or api.deps.get_detector
        app.state.detector_lock = threading.Lock()

        def _prewarm_detector() -> None:
            """Absorb the ~0.4s model load at startup instead of the first scan."""
            from glassfit.api.deps import mediapipe_available
            from glassfit.vision.tasks_landmarker import TasksLandmarkerBackend

            try:
                if mediapipe_available(app_settings):
                    detector = TasksLandmarkerBackend(app_settings.landmarker_model_path)
                    with app.state.detector_lock:
                        if app.state.detector is None:
                            app.state.detector = detector
            except Exception:  # noqa: BLE001 - prewarm is best-effort; the lazy path reports
                pass

        threading.Thread(target=_prewarm_detector, name="detector-prewarm", daemon=True).start()
        yield
        conn.close()

    app = FastAPI(title="GlassFit", version=__version__, lifespan=lifespan)
    register_error_handlers(app)
    for router_module in (health, scan, measurements, recommendations, frames, feedback):
        app.include_router(router_module.router, prefix="/api/v1")
    # Optional local try-on art (personal-use product photos; gitignored).
    if app_settings.tryon_dir.is_dir():
        app.mount("/tryon", StaticFiles(directory=app_settings.tryon_dir))
    # Mounted LAST so /api/v1/* and /tryon always win over the static catch-all.
    if app_settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=app_settings.frontend_dir, html=True))
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("glassfit.app:app", host=settings.host, port=settings.port, reload=settings.dev)
