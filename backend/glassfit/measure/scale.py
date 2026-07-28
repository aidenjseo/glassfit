"""Scale + side resolution: normalized landmarks -> isotropic units -> millimeters.

Sides are SUBJECT-anatomical (right = the subject's right eye = OD). In an unmirrored
camera frame the subject's right eye has the SMALLER x coordinate — sides are resolved
at runtime from the data, never trusted from index-table labels.
"""

from dataclasses import dataclass

import numpy as np

from ..schemas import LandmarkSet
from . import indices as idx
from .geometry import dist, midpoint

PD_SCALE_FALLBACK_WARNING = "pd_scale_fallback_eye_corners"

__all__ = [
    "PD_SCALE_FALLBACK_WARNING",
    "ResolvedSides",
    "SideIndices",
    "mm_per_unit",
    "resolve_sides",
    "to_unit_space",
]


def to_unit_space(ls: LandmarkSet) -> np.ndarray:
    """Map normalized landmarks to isotropic pixel units: (x*W, y*H, z*W) -> (N, 3).

    MediaPipe normalizes x by image width, y by image height and z roughly on the x
    scale; multiplying back out yields a space where one unit is one pixel in every
    axis, so Euclidean distances are meaningful.
    """
    pts = np.asarray(ls.points, dtype=float)
    scale = np.array([ls.image_width, ls.image_height, ls.image_width], dtype=float)
    return pts * scale


@dataclass(frozen=True)
class SideIndices:
    """Landmark indices for one anatomical side, as used by the extractor."""

    iris_center: int
    iris_ring: tuple[int, ...]
    outer_canthus: int
    inner_canthus: int
    upper_lid: int
    lower_lid: int
    mid_cheek: int
    face_side: int
    temple: int
    jaw: int


_CANONICAL_RIGHT = SideIndices(
    iris_center=idx.IRIS_CENTER_RIGHT,
    iris_ring=idx.IRIS_RING_RIGHT,
    outer_canthus=idx.OUTER_CANTHUS_RIGHT,
    inner_canthus=idx.INNER_CANTHUS_RIGHT,
    upper_lid=idx.UPPER_LID_RIGHT,
    lower_lid=idx.LOWER_LID_RIGHT,
    mid_cheek=idx.MID_CHEEK_RIGHT,
    face_side=idx.FACE_SIDE_RIGHT,
    temple=idx.TEMPLE_RIGHT,
    jaw=idx.JAW_RIGHT[0],
)
_CANONICAL_LEFT = SideIndices(
    iris_center=idx.IRIS_CENTER_LEFT,
    iris_ring=idx.IRIS_RING_LEFT,
    outer_canthus=idx.OUTER_CANTHUS_LEFT,
    inner_canthus=idx.INNER_CANTHUS_LEFT,
    upper_lid=idx.UPPER_LID_LEFT,
    lower_lid=idx.LOWER_LID_LEFT,
    mid_cheek=idx.MID_CHEEK_LEFT,
    face_side=idx.FACE_SIDE_LEFT,
    temple=idx.TEMPLE_LEFT,
    jaw=idx.JAW_LEFT[0],
)


@dataclass(frozen=True)
class ResolvedSides:
    """Subject-anatomical side assignment resolved from the frame's x coordinates."""

    right: SideIndices
    left: SideIndices
    flipped: bool  # True when the frame contradicts the canonical index-table labels


def resolve_sides(pts: np.ndarray) -> ResolvedSides:
    """Assign subject right/left landmark index groups by x coordinate.

    Uses the mean x of each eye's canthi (present in both 468- and 478-point sets).
    In an unmirrored frame the subject-right eye has the smaller x; if the canonical
    "right"-labeled eye sits at larger x the frame is mirrored and the groups swap.
    ``pts`` may be in any space whose x axis matches the image (normalized, unit, mm).
    """
    right_x = float(pts[idx.OUTER_CANTHUS_RIGHT, 0] + pts[idx.INNER_CANTHUS_RIGHT, 0]) / 2.0
    left_x = float(pts[idx.OUTER_CANTHUS_LEFT, 0] + pts[idx.INNER_CANTHUS_LEFT, 0]) / 2.0
    if right_x > left_x:
        return ResolvedSides(right=_CANONICAL_LEFT, left=_CANONICAL_RIGHT, flipped=True)
    return ResolvedSides(right=_CANONICAL_RIGHT, left=_CANONICAL_LEFT, flipped=False)


def mm_per_unit(pd_mm: float, pts: np.ndarray, has_iris: bool) -> tuple[float, list[str]]:
    """Millimeters per isotropic unit, anchored on the user-entered PD.

    With iris landmarks (478-point set) the scale is ``pd_mm`` divided by the
    iris-center distance (468<->473). Without them (468-point set) it falls back to the
    distance between the eye-corner midpoints — a slight PD underestimate, flagged with
    the warning ``pd_scale_fallback_eye_corners``.

    ``pts`` must be in the isotropic unit space from :func:`to_unit_space`.
    Side labels do not matter here: the pupil-to-pupil distance is symmetric.
    """
    warnings: list[str] = []
    if has_iris and len(pts) >= 478:
        pd_units = dist(pts[idx.IRIS_CENTER_RIGHT], pts[idx.IRIS_CENTER_LEFT])
    else:
        eye_a = midpoint(pts[idx.OUTER_CANTHUS_RIGHT], pts[idx.INNER_CANTHUS_RIGHT])
        eye_b = midpoint(pts[idx.OUTER_CANTHUS_LEFT], pts[idx.INNER_CANTHUS_LEFT])
        pd_units = dist(eye_a, eye_b)
        warnings.append(PD_SCALE_FALLBACK_WARNING)
    # `not (x > 0)` (rather than `x <= 0`) also catches NaN, whose comparisons are False
    if not pd_units > 0.0:
        raise ValueError("degenerate landmarks: zero or non-finite interpupillary distance")
    return pd_mm / pd_units, warnings
