"""Backend-agnostic landmark-detection primitives.

Pure numpy — this module must import without mediapipe/cv2 installed.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class DetectionResult:
    """Result of running a landmark backend on a single frame.

    Attributes:
        landmarks: ``(478, 3)`` float array of the FIRST detected face in NORMALIZED image
            coordinates (x in [0, 1] of width, y in [0, 1] of height, z roughly x-scaled,
            per the MediaPipe convention). Shape ``(0, 3)`` when no face was detected.
        face_count: number of faces the backend detected in the frame (may exceed 1 even
            though only the first face's landmarks are returned).
        image_width: source frame width in pixels.
        image_height: source frame height in pixels.
        head_pose_deg: ``(yaw, pitch, roll)`` in DEGREES from the backend's head-pose
            estimate (e.g. the facial transformation matrix), or ``None`` when unavailable —
            callers should then fall back to a geometric estimate.
    """

    landmarks: np.ndarray
    face_count: int
    image_width: int
    image_height: int
    head_pose_deg: tuple[float, float, float] | None


class LandmarkBackend(Protocol):
    """Anything that turns a BGR frame into a DetectionResult."""

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        """Run detection on an ``(H, W, 3)`` uint8 BGR frame."""
        ...
