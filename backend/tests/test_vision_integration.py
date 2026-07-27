"""Real-model integration tests. Excluded by default (addopts = -m 'not mediapipe').

Run with: uv run pytest backend/tests/test_vision_integration.py -m mediapipe
Skips cleanly when mediapipe is not installed or the .task model has not been downloaded.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.mediapipe


def _backend_or_skip():
    pytest.importorskip("mediapipe")
    from glassfit.config import get_settings
    from glassfit.vision.tasks_landmarker import TasksLandmarkerBackend

    model_path = get_settings().landmarker_model_path
    if not model_path.is_file():
        pytest.skip(
            f"model not downloaded at {model_path}; run: uv run python scripts/download_models.py"
        )
    return TasksLandmarkerBackend(model_path)


def test_blank_image_yields_no_face_but_well_formed_result() -> None:
    backend = _backend_or_skip()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)  # BGR

    det = backend.detect(blank)

    assert det.face_count == 0
    assert det.landmarks.shape == (0, 3)
    assert det.image_width == 640
    assert det.image_height == 480
    assert det.head_pose_deg is None


def test_missing_model_raises_actionable_runtime_error(tmp_path) -> None:
    from glassfit.vision.tasks_landmarker import TasksLandmarkerBackend

    with pytest.raises(RuntimeError, match="download_models"):
        TasksLandmarkerBackend(tmp_path / "does_not_exist.task")
