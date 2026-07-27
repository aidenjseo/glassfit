"""Shared test fixtures: synthetic landmarks, fake detector, app/client factories."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from glassfit.api.deps import get_detector
from glassfit.app import create_app
from glassfit.config import Settings
from glassfit.schemas import LandmarkSet
from glassfit.vision.base import DetectionResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_landmark_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / "landmarks" / f"{name}.json").read_text())


def load_landmark_set(name: str) -> tuple[LandmarkSet, dict]:
    """Fixture as a validated LandmarkSet plus the full payload (expected values etc.)."""
    payload = load_landmark_fixture(name)
    return LandmarkSet(**payload["landmark_set"]), payload


def truncate_to_468(ls: LandmarkSet) -> LandmarkSet:
    """A copy without the 10 iris points — the no-iris (legacy-mesh) shape."""
    return LandmarkSet(
        points=ls.points[:468], image_width=ls.image_width, image_height=ls.image_height
    )


class FakeBackend:
    """Detector stand-in returning fixture landmarks — no mediapipe required.

    Pass ``script`` (a list of ``{"fixture", "pose", "face_count"}`` dicts) to vary
    the result per ``detect()`` call, e.g. a frontal burst followed by head-turn
    frames; the last entry repeats once the script is exhausted.
    """

    def __init__(
        self,
        fixture: str = "synthetic_average",
        face_count: int = 1,
        pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        script: list[dict] | None = None,
    ):
        self._default = {"fixture": fixture, "face_count": face_count, "pose": pose}
        self._script = list(script) if script else None
        self._calls = 0
        self._cache: dict[str, tuple[np.ndarray, int, int]] = {}

    def _load(self, fixture: str) -> tuple[np.ndarray, int, int]:
        if fixture not in self._cache:
            data = load_landmark_fixture(fixture)["landmark_set"]
            self._cache[fixture] = (
                np.asarray(data["points"], dtype=float),
                data["image_width"],
                data["image_height"],
            )
        return self._cache[fixture]

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        if self._script:
            spec = self._script[min(self._calls, len(self._script) - 1)]
        else:
            spec = self._default
        self._calls += 1
        merged = {**self._default, **spec}
        landmarks, width, height = self._load(merged["fixture"])
        face_count = merged["face_count"]
        return DetectionResult(
            landmarks=landmarks if face_count >= 1 else np.zeros((0, 3)),
            face_count=face_count,
            image_width=width,
            image_height=height,
            head_pose_deg=tuple(merged["pose"]),
        )


@pytest.fixture
def make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., FastAPI]:
    """App factory on a temp DB; cv2-free decode; optional fake detector."""
    from glassfit.services import scan_service

    monkeypatch.setattr(
        scan_service, "decode_image", lambda blob: np.zeros((720, 1280, 3), dtype=np.uint8)
    )

    def _make(detector: object | None = None, **settings_overrides: object) -> FastAPI:
        defaults: dict[str, object] = {
            "db_path": tmp_path / "data" / "glassfit.db",
            "save_frames_dir": tmp_path / "data" / "frames",
            "frontend_dir": tmp_path / "no-frontend",
            # A missing model path keeps the lifespan detector-prewarm inert in tests —
            # otherwise every TestClient app loads a real mediapipe graph in a thread,
            # which wedges the interpreter at exit.
            "landmarker_model_path": tmp_path / "no-model.task",
        }
        settings = Settings(**{**defaults, **settings_overrides})
        app = create_app(settings)
        if detector is not None:
            app.dependency_overrides[get_detector] = lambda: detector
        return app

    return _make


@pytest.fixture
def client(make_app: Callable[..., FastAPI]):
    with TestClient(make_app(detector=FakeBackend())) as test_client:
        yield test_client


def jpeg_upload_files(n: int = 6) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Multipart file tuples; content is irrelevant (decode is patched in tests)."""
    return [("frames", (f"frame_{i}.jpg", b"fake-jpeg-bytes", "image/jpeg")) for i in range(n)]
