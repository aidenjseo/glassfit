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
from glassfit.vision.base import DetectionResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_landmark_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / "landmarks" / f"{name}.json").read_text())


class FakeBackend:
    """Detector stand-in returning fixture landmarks — no mediapipe required."""

    def __init__(
        self,
        fixture: str = "synthetic_average",
        face_count: int = 1,
        pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        data = load_landmark_fixture(fixture)["landmark_set"]
        self._landmarks = np.asarray(data["points"], dtype=float)
        self._w: int = data["image_width"]
        self._h: int = data["image_height"]
        self.face_count = face_count
        self.pose = pose

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        landmarks = self._landmarks if self.face_count >= 1 else np.zeros((0, 3))
        return DetectionResult(
            landmarks=landmarks,
            face_count=self.face_count,
            image_width=self._w,
            image_height=self._h,
            head_pose_deg=self.pose,
        )


@pytest.fixture
def make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., FastAPI]:
    """App factory on a temp DB; cv2-free decode; optional fake detector."""
    from glassfit.services import scan_service

    monkeypatch.setattr(
        scan_service, "decode_image", lambda blob: np.zeros((720, 1280, 3), dtype=np.uint8)
    )

    def _make(detector: object | None = None, **settings_overrides: object) -> FastAPI:
        settings = Settings(
            db_path=tmp_path / "data" / "glassfit.db",
            save_frames_dir=tmp_path / "data" / "frames",
            frontend_dir=tmp_path / "no-frontend",
            **settings_overrides,
        )
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
