"""FastAPI dependency providers (overridable in tests via app.dependency_overrides)."""

import importlib.util
from typing import Annotated

from fastapi import Depends, Request

from glassfit.config import Settings
from glassfit.errors import MediapipeUnavailable
from glassfit.storage.repo import Repo
from glassfit.vision.base import LandmarkBackend
from glassfit.vision.tasks_landmarker import TasksLandmarkerBackend


def mediapipe_available(settings: Settings) -> bool:
    return (
        importlib.util.find_spec("mediapipe") is not None
        and settings.landmarker_model_path.exists()
    )


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_repo(request: Request) -> Repo:
    return request.app.state.repo


def get_detector(request: Request) -> LandmarkBackend:
    """Lazily build the (expensive, non-thread-safe) landmarker once per app."""
    state = request.app.state
    # Fast path: once built, don't contend with in-flight inference for the lock.
    if state.detector is not None:
        return state.detector
    with state.detector_lock:
        if state.detector is None:
            settings: Settings = state.settings
            if importlib.util.find_spec("mediapipe") is None:
                raise MediapipeUnavailable(
                    "mediapipe is not installed — run `uv sync --extra scan`"
                )
            if not settings.landmarker_model_path.exists():
                raise MediapipeUnavailable(
                    f"face landmarker model missing at {settings.landmarker_model_path} — "
                    "run `uv run python scripts/download_models.py`"
                )
            state.detector = TasksLandmarkerBackend(settings.landmarker_model_path)
    return state.detector


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
RepoDep = Annotated[Repo, Depends(get_repo)]
DetectorDep = Annotated[LandmarkBackend, Depends(get_detector)]
