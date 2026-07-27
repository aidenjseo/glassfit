"""Ear-region estimates from head-turn (side-view) frames.

The frontal mesh has no ear landmarks and its z at the face boundary is weak, so the
frontal hinge-to-ear value is a crude proxy. A turned head exposes one side of the
face almost in-plane, where the outer-canthus -> pre-auricular distance is barely
foreshortened — a far better base for the hinge-to-ear estimate.

Scale WITHOUT the user's PD: in a yawed frame the interpupillary distance is
foreshortened by cos(yaw), but the human iris diameter is nearly constant at
~11.7 mm (the basis of MediaPipe Iris depth estimation), so the exposed (near)
eye's iris ring gives a per-frame absolute scale.
"""

import numpy as np

from glassfit.schemas import LandmarkSet

from .extractor import TRAGUS_BEHIND_FACE_OVAL_MM
from .geometry import dist
from .scale import SideIndices, resolve_sides, to_unit_space

IRIS_DIAMETER_MM = 11.7


def _iris_diameter_units(pts: np.ndarray, side: SideIndices) -> float:
    ring = pts[list(side.iris_ring)][:, :2]  # xy only: ring z is noisy
    diam = 0.0
    for i in range(len(ring)):
        for j in range(i + 1, len(ring)):
            diam = max(diam, dist(ring[i], ring[j]))
    return diam


def hinge_to_ear_from_side_view(landmarks: LandmarkSet) -> tuple[str, float] | None:
    """One side-view frame -> (exposed subject side, hinge-to-ear estimate in mm).

    The exposed side is decided GEOMETRICALLY (the face-side landmark closer to the
    camera, i.e. smaller z), never from turn-direction labels. Returns None when the
    frame lacks iris landmarks or the scale degenerates.
    """
    if not landmarks.has_iris:
        return None
    pts = to_unit_space(landmarks)
    sides = resolve_sides(pts)
    exposed_is_right = pts[sides.right.face_side, 2] < pts[sides.left.face_side, 2]
    exposed = sides.right if exposed_is_right else sides.left
    diam = _iris_diameter_units(pts, exposed)
    if diam <= 1e-9:
        return None
    scale = IRIS_DIAMETER_MM / diam
    canthus_to_side = dist(pts[exposed.outer_canthus], pts[exposed.face_side]) * scale
    estimate = canthus_to_side + TRAGUS_BEHIND_FACE_OVAL_MM
    # sanity band: reject degenerate geometry rather than poison the estimate
    if not 40.0 <= estimate <= 140.0:
        return None
    return ("right" if exposed_is_right else "left"), estimate


def summarize_side_views(
    side_sets: list[LandmarkSet],
) -> tuple[dict[str, int], dict[str, float | None]]:
    """Median per-side hinge-to-ear estimates from all usable side-view frames."""
    per_side: dict[str, list[float]] = {"right": [], "left": []}
    for landmarks in side_sets:
        result = hinge_to_ear_from_side_view(landmarks)
        if result is not None:
            side, value = result
            per_side[side].append(value)
    counts = {side: len(vals) for side, vals in per_side.items()}
    medians: dict[str, float | None] = {
        side: (float(np.median(vals)) if vals else None) for side, vals in per_side.items()
    }
    return counts, medians
