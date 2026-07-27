"""Tests for glassfit.vision.quality (pure numpy, no mediapipe)."""

import numpy as np
import pytest

from glassfit.measure.indices import CHIN, FACE_SIDE_LEFT, FACE_SIDE_RIGHT, NASION
from glassfit.vision.base import DetectionResult
from glassfit.vision.quality import estimate_pose_deg, evaluate_frame

GATES = {
    "max_yaw_deg": 12.0,
    "max_pitch_deg": 12.0,
    "max_roll_deg": 15.0,
    "min_face_width_frac": 0.30,
}


def _frontal_landmarks() -> np.ndarray:
    """Synthetic frontal-face normalized landmarks with the pose-relevant indices pinned."""
    rng = np.random.default_rng(7)
    pts = rng.uniform([0.32, 0.25, -0.03], [0.68, 0.75, 0.03], size=(478, 3))
    pts[NASION] = (0.5, 0.40, -0.02)
    pts[CHIN] = (0.5, 0.65, -0.02)
    pts[FACE_SIDE_RIGHT] = (0.30, 0.50, 0.0)  # subject-right: smaller x (unmirrored frame)
    pts[FACE_SIDE_LEFT] = (0.70, 0.50, 0.0)
    return pts


def _det(
    landmarks: np.ndarray,
    face_count: int = 1,
    head_pose_deg: tuple[float, float, float] | None = None,
) -> DetectionResult:
    return DetectionResult(
        landmarks=landmarks,
        face_count=face_count,
        image_width=640,
        image_height=480,
        head_pose_deg=head_pose_deg,
    )


# --- estimate_pose_deg -------------------------------------------------------------------


def test_estimate_pose_frontal_is_near_zero() -> None:
    yaw, pitch, roll = estimate_pose_deg(_frontal_landmarks())
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(roll) < 1e-6


def test_estimate_pose_detects_yaw_via_side_z_asymmetry() -> None:
    pts = _frontal_landmarks()
    pts[FACE_SIDE_RIGHT] = (0.30, 0.50, 0.10)  # subject-right side farther from camera
    pts[FACE_SIDE_LEFT] = (0.70, 0.50, -0.10)
    yaw, _, _ = estimate_pose_deg(pts)
    assert yaw > 20.0  # atan2(0.2, 0.4) ~ 26.6 deg, positive = turned to subject's right


def test_estimate_pose_resolves_sides_by_x_not_by_index() -> None:
    pts = _frontal_landmarks()
    # Swap positions of the two side landmarks: index labels now contradict x order.
    pts[FACE_SIDE_RIGHT] = (0.70, 0.50, -0.10)
    pts[FACE_SIDE_LEFT] = (0.30, 0.50, 0.10)
    yaw, _, _ = estimate_pose_deg(pts)
    assert yaw > 20.0  # same face geometry -> same yaw, regardless of which index holds it


def test_estimate_pose_detects_pitch_and_roll() -> None:
    pitched = _frontal_landmarks()
    pitched[CHIN] = (0.5, 0.65, -0.12)  # chin toward camera -> looking up -> positive pitch
    _, pitch, _ = estimate_pose_deg(pitched)
    assert pitch > 15.0

    rolled = _frontal_landmarks()
    rolled[CHIN] = (0.6, 0.65, -0.02)  # chin displaced toward +x -> positive roll
    _, _, roll = estimate_pose_deg(rolled)
    assert roll > 15.0


# --- evaluate_frame ----------------------------------------------------------------------


def test_frontal_frame_accepted() -> None:
    det = _det(_frontal_landmarks(), head_pose_deg=(0.0, 0.0, 0.0))
    report = evaluate_frame(det, **GATES, index=3)
    assert report.accepted is True
    assert report.reject_reason is None
    assert report.index == 3
    assert report.yaw_deg == 0.0
    assert report.pitch_deg == 0.0
    assert report.roll_deg == 0.0


def test_frontal_frame_accepted_via_geometric_fallback() -> None:
    report = evaluate_frame(_det(_frontal_landmarks(), head_pose_deg=None), **GATES, index=0)
    assert report.accepted is True
    assert report.yaw_deg is not None  # fallback estimate was filled in


def test_no_face_rejected() -> None:
    det = _det(np.empty((0, 3)), face_count=0)
    report = evaluate_frame(det, **GATES, index=1)
    assert report.accepted is False
    assert report.reject_reason == "no_face"
    assert report.yaw_deg is None


def test_multiple_faces_rejected() -> None:
    report = evaluate_frame(
        _det(_frontal_landmarks(), face_count=2, head_pose_deg=(0.0, 0.0, 0.0)), **GATES, index=2
    )
    assert report.accepted is False
    assert report.reject_reason == "multiple_faces"


@pytest.mark.parametrize(
    ("pose", "reason"),
    [
        ((20.0, 0.0, 0.0), "yaw_exceeded"),
        ((-20.0, 0.0, 0.0), "yaw_exceeded"),
        ((0.0, 20.0, 0.0), "pitch_exceeded"),
        ((0.0, -20.0, 0.0), "pitch_exceeded"),
        ((0.0, 0.0, 20.0), "roll_exceeded"),
        ((0.0, 0.0, -20.0), "roll_exceeded"),
    ],
)
def test_backend_pose_over_limit_rejected(pose: tuple[float, float, float], reason: str) -> None:
    report = evaluate_frame(_det(_frontal_landmarks(), head_pose_deg=pose), **GATES, index=0)
    assert report.accepted is False
    assert report.reject_reason == reason


def test_yawed_face_rejected_via_geometric_fallback() -> None:
    pts = _frontal_landmarks()
    pts[FACE_SIDE_RIGHT] = (0.30, 0.50, 0.10)
    pts[FACE_SIDE_LEFT] = (0.70, 0.50, -0.10)
    report = evaluate_frame(_det(pts, head_pose_deg=None), **GATES, index=0)
    assert report.accepted is False
    assert report.reject_reason == "yaw_exceeded"


def test_small_face_rejected() -> None:
    pts = _frontal_landmarks()
    pts[:, 0] = 0.5 + (pts[:, 0] - 0.5) * 0.25  # shrink x-extent to ~0.1 of image width
    report = evaluate_frame(_det(pts, head_pose_deg=(0.0, 0.0, 0.0)), **GATES, index=5)
    assert report.accepted is False
    assert report.reject_reason == "face_too_small"
