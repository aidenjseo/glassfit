"""MediaPipe Tasks FaceLandmarker backend.

The ONLY module that touches mediapipe, and only lazily inside ``__init__`` — importing this
module must succeed without mediapipe installed.
"""

import math
from pathlib import Path

import numpy as np

from .base import DetectionResult

_EMPTY_LANDMARKS = np.empty((0, 3), dtype=np.float64)


def _rotation_to_euler_deg(rot: np.ndarray) -> tuple[float, float, float]:
    """Decompose a 3x3 rotation matrix into (yaw, pitch, roll) degrees.

    Convention: the facial transformation matrix maps the canonical face model into MediaPipe's
    camera space (right-handed: +X to the viewer's right, +Y up, +Z toward the camera; the
    canonical face looks along +Z). We decompose R = Rz(roll) @ Ry(yaw) @ Rx(pitch)
    (intrinsic Tait-Bryan Z-Y-X), so:

    - yaw   = rotation about the camera's vertical (Y) axis — head turned left/right,
    - pitch = rotation about the camera's horizontal (X) axis — head nodded up/down,
    - roll  = rotation about the optical (Z) axis — in-plane head tilt.

    Quality gating uses magnitudes only; signs follow the camera axes above.
    """
    sy = math.hypot(rot[0, 0], rot[1, 0])
    if sy < 1e-8:  # gimbal lock: |yaw| ~ 90 deg, roll/pitch are not separable
        yaw = math.copysign(math.pi / 2.0, -rot[2, 0])
        pitch = math.atan2(-rot[1, 2], rot[1, 1])
        roll = 0.0
    else:
        yaw = math.atan2(-rot[2, 0], sy)
        pitch = math.atan2(rot[2, 1], rot[2, 2])
        roll = math.atan2(rot[1, 0], rot[0, 0])
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


class TasksLandmarkerBackend:
    """FaceLandmarker (MediaPipe Tasks API, IMAGE running mode) landmark backend.

    NOT thread-safe: the underlying FaceLandmarker keeps mutable graph state, so callers MUST
    serialize all ``detect`` calls on one instance (e.g. hold a ``threading.Lock``).
    """

    def __init__(self, model_path: Path):
        if not Path(model_path).is_file():
            raise RuntimeError(
                f"FaceLandmarker model not found at {model_path}. "
                "Download it with: uv run python scripts/download_models.py"
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=2,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        """Detect face landmarks on an ``(H, W, 3)`` uint8 BGR frame."""
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) BGR image, got shape {bgr.shape}")
        height, width = int(bgr.shape[0]), int(bgr.shape[1])

        mp = self._mp
        rgb = np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.uint8)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)

        face_count = len(result.face_landmarks)
        if face_count == 0:
            return DetectionResult(
                landmarks=_EMPTY_LANDMARKS.copy(),
                face_count=0,
                image_width=width,
                image_height=height,
                head_pose_deg=None,
            )

        first = result.face_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in first], dtype=np.float64)

        head_pose_deg: tuple[float, float, float] | None = None
        matrixes = result.facial_transformation_matrixes
        if matrixes:
            matrix = np.asarray(matrixes[0], dtype=np.float64)
            head_pose_deg = _rotation_to_euler_deg(matrix[:3, :3])

        return DetectionResult(
            landmarks=landmarks,
            face_count=face_count,
            image_width=width,
            image_height=height,
            head_pose_deg=head_pose_deg,
        )
