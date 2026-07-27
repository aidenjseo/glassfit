"""Per-frame quality gating: head pose + face size checks. Pure numpy — NO mediapipe."""

import math

import numpy as np

from glassfit.measure.indices import CHIN, FACE_SIDE_LEFT, FACE_SIDE_RIGHT, NASION
from glassfit.schemas import FrameReport

from .base import DetectionResult


def estimate_pose_deg(pts: np.ndarray) -> tuple[float, float, float]:
    """Geometric (yaw, pitch, roll) estimate in DEGREES from normalized landmarks ``(N, 3)``.

    Fallback for when the backend supplies no head pose. Method:

    - roll:  in-plane angle of the nasion->chin vector vs the image vertical; positive when the
      chin is displaced toward larger x relative to the nasion (unmirrored frame).
    - pitch: out-of-plane angle of the nasion->chin vector (z vs in-plane length); positive when
      looking up (chin closer to the camera, i.e. more-negative z relative to the nasion).
    - yaw:   z-asymmetry of the lateral face-oval points (234/454) vs their x-extent; positive
      when the subject turns toward their own right (subject-right side farther from camera).
      Sides are resolved at runtime by x-coordinate: in an unmirrored frame the SUBJECT-right
      side has the SMALLER x — never trust index-table side labels.

    Limits (coarse gating only, NOT for measurement):
    - x/y are normalized by width/height, so non-square aspect ratios distort roll/pitch;
    - z uses MediaPipe's "roughly x-scaled" convention, another first-order approximation;
    - anatomy biases it: nasion and chin do not lie at identical depth on a truly frontal face,
      so a small constant pitch offset is expected;
    - no perspective model. Expect several degrees of error; fine for ~12-15 deg thresholds.
    """
    nasion = pts[NASION]
    chin = pts[CHIN]
    dx = float(chin[0] - nasion[0])
    dy = float(chin[1] - nasion[1])  # image y grows downward; chin is below nasion -> dy > 0
    dz = float(chin[2] - nasion[2])

    roll = math.degrees(math.atan2(dx, dy))
    pitch = math.degrees(math.atan2(-dz, math.hypot(dx, dy)))

    # Deliberately NOT measure.scale.resolve_sides here: this coarse estimator must work
    # on partial landmark arrays (only 4 indices populated in gating tests), and ordering
    # exactly two points by x IS the codebase's side convention, applied locally.
    side_a = pts[FACE_SIDE_RIGHT]
    side_b = pts[FACE_SIDE_LEFT]
    right, left = (side_a, side_b) if side_a[0] <= side_b[0] else (side_b, side_a)
    face_width = float(left[0] - right[0])
    yaw = math.degrees(math.atan2(float(right[2] - left[2]), face_width))

    return (yaw, pitch, roll)


def pose_of(det: DetectionResult) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees: the backend's head pose, else the geometric estimate."""
    if det.head_pose_deg is not None:
        return det.head_pose_deg
    return estimate_pose_deg(det.landmarks)


def evaluate_frame(
    det: DetectionResult,
    *,
    max_yaw_deg: float,
    max_pitch_deg: float,
    max_roll_deg: float,
    min_face_width_frac: float,
    index: int,
) -> FrameReport:
    """Gate a single frame's detection, returning a FrameReport.

    ``reject_reason`` is one of: "no_face", "multiple_faces", "yaw_exceeded", "pitch_exceeded",
    "roll_exceeded", "face_too_small" (first failing check in that order wins). Uses
    ``det.head_pose_deg`` when the backend provided it, else the geometric estimate. Face width
    fraction is the landmark x-extent (normalized x is already a fraction of image width).
    """
    if det.face_count == 0 or det.landmarks.size == 0:
        return FrameReport(index=index, accepted=False, reject_reason="no_face")

    yaw, pitch, roll = pose_of(det)

    reject_reason: str | None = None
    if det.face_count > 1:
        reject_reason = "multiple_faces"
    elif abs(yaw) > max_yaw_deg:
        reject_reason = "yaw_exceeded"
    elif abs(pitch) > max_pitch_deg:
        reject_reason = "pitch_exceeded"
    elif abs(roll) > max_roll_deg:
        reject_reason = "roll_exceeded"
    else:
        xs = det.landmarks[:, 0]
        if float(xs.max() - xs.min()) < min_face_width_frac:
            reject_reason = "face_too_small"

    return FrameReport(
        index=index,
        accepted=reject_reason is None,
        reject_reason=reject_reason,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
    )
