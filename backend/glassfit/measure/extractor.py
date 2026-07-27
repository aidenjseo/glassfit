"""Measurement extraction: LandmarkSet + entered PD -> FaceMeasurements.

Pure numpy + schemas — this module must never import mediapipe or fastapi.

Conventions
-----------
* Every field function operates on ``mm``: the (N, 3) landmark array already converted
  to millimeters (isotropic unit space times the PD-anchored scale), image axes:
  x grows toward the subject's LEFT, y grows DOWN, z grows AWAY from the camera.
* Sides are SUBJECT-anatomical and come from :func:`glassfit.measure.scale.resolve_sides`
  (resolved by x coordinate, never by index-table label).
* The face mesh has NO ear landmarks: hinge_to_ear_mm, ear_height_asymmetry_mm and
  behind_ear are ESTIMATES built from the FACE_SIDE (pre-auricular face-oval) proxies
  plus the documented anthropometric offsets below, and are always listed — together
  with vertex_estimate_mm — in ``quality.low_confidence_fields``.
"""

import numpy as np

from ..schemas import (
    BehindEar,
    BridgeWidths,
    FaceMeasurements,
    LandmarkSet,
    MeasurementQuality,
    PerSide,
)
from . import indices as idx
from .geometry import (
    angle_to_horizontal_deg,
    dist,
    fit_circle_radius,
    midpoint,
    polyline_station,
)
from .scale import ResolvedSides, SideIndices, mm_per_unit, resolve_sides, to_unit_space

# --- Anthropometric constants for the ear-region ESTIMATES (no ear landmarks exist) ---
# The tragus sits roughly 15-20 mm behind the lateral face-oval edge (234/454).
TRAGUS_BEHIND_FACE_OVAL_MM = 17.5
# Total temple length ~ hinge-to-ear + 25-35 mm bend allowance (exported for the rules engine).
TEMPLE_BEND_ALLOWANCE_MM = 30.0
# Fraction of the face-side -> gonion vertical gap that the temple tip drops behind the ear.
BEHIND_EAR_DROP_FRACTION = 0.4
# Horizontal run over which the behind-ear bend develops (sets the bend angle).
BEHIND_EAR_BEND_RUN_MM = 25.0

# Implied zygoma width outside this range means the entered PD is probably wrong.
ZYGOMA_PLAUSIBLE_MM: tuple[float, float] = (100.0, 165.0)

LOW_CONFIDENCE_FIELDS: tuple[str, ...] = (
    "hinge_to_ear_mm",
    "ear_height_asymmetry_mm",
    "behind_ear",
    "vertex_estimate_mm",
)

__all__ = [
    "BEHIND_EAR_BEND_RUN_MM",
    "BEHIND_EAR_DROP_FRACTION",
    "LOW_CONFIDENCE_FIELDS",
    "TEMPLE_BEND_ALLOWANCE_MM",
    "TRAGUS_BEHIND_FACE_OVAL_MM",
    "ZYGOMA_PLAUSIBLE_MM",
    "behind_ear_estimate",
    "binocular_pd_mm",
    "bridge_crest_height_mm",
    "bridge_widths_mm",
    "canthal_tilt_deg",
    "cheek_clearance_mm",
    "compute_measurements",
    "ear_height_asymmetry_mm",
    "face_wrap_radius_mm",
    "hinge_to_ear_mm",
    "monocular_pd_mm",
    "pupil_height_ratio",
    "temple_width_mm",
    "vertex_estimate_mm",
    "zygoma_width_mm",
]


def _pupil_centers(
    mm: np.ndarray, sides: ResolvedSides, has_iris: bool
) -> tuple[np.ndarray, np.ndarray]:
    """(subject-right, subject-left) pupil centers: iris centers, or eye-corner midpoints
    as a proxy on 468-point sets."""
    if has_iris and len(mm) >= 478:
        return mm[sides.right.iris_center], mm[sides.left.iris_center]
    right = midpoint(mm[sides.right.outer_canthus], mm[sides.right.inner_canthus])
    left = midpoint(mm[sides.left.outer_canthus], mm[sides.left.inner_canthus])
    return right, left


def binocular_pd_mm(mm: np.ndarray, sides: ResolvedSides, has_iris: bool) -> float:
    """Binocular PD: 3-D distance between the pupil centers."""
    right, left = _pupil_centers(mm, sides, has_iris)
    return dist(right, left)


def monocular_pd_mm(mm: np.ndarray, sides: ResolvedSides, has_iris: bool) -> PerSide:
    """Monocular PDs: each pupil's distance from the nasion (168) midline.

    Measured along the interpupillary axis (xy projection) so a rolled head does not
    corrupt the split; the two sides sum to the projected binocular PD.
    """
    right, left = _pupil_centers(mm, sides, has_iris)
    nasion = mm[idx.NASION]
    axis = (left - right)[:2]
    norm = float(np.linalg.norm(axis))
    u = axis / norm if norm > 0.0 else np.array([1.0, 0.0])
    return PerSide(
        right=abs(float(np.dot((right - nasion)[:2], u))),
        left=abs(float(np.dot((left - nasion)[:2], u))),
    )


def bridge_widths_mm(mm: np.ndarray) -> BridgeWidths:
    """Nose-bridge widths at the crest and 10/15 mm below it.

    Each sidewall pair in ``indices.BRIDGE_SIDEWALL_PAIRS`` yields one (depth, width)
    sample: width is the 3-D distance across the pair, depth is the arc-length station
    of the pair's midpoint along the nose-dorsum chain (crest = station 0 at nasion).
    The three requested depths are linearly interpolated between samples and clamped to
    the sampled range (so "at crest" reports the shallowest sampled width).
    Sidewall pairs are symmetric, so side resolution is irrelevant here.
    """
    chain = mm[list(idx.NOSE_DORSUM)]
    depths: list[float] = []
    widths: list[float] = []
    for pair_a, pair_b in idx.BRIDGE_SIDEWALL_PAIRS:
        widths.append(dist(mm[pair_a], mm[pair_b]))
        depths.append(polyline_station(chain, midpoint(mm[pair_a], mm[pair_b])))
    order = np.argsort(depths)
    depths_arr = np.asarray(depths)[order]
    widths_arr = np.asarray(widths)[order]
    return BridgeWidths(
        at_crest_mm=float(np.interp(0.0, depths_arr, widths_arr)),
        below_10mm_mm=float(np.interp(10.0, depths_arr, widths_arr)),
        below_15mm_mm=float(np.interp(15.0, depths_arr, widths_arr)),
    )


def bridge_crest_height_mm(mm: np.ndarray, sides: ResolvedSides) -> float:
    """Nasion prominence in z relative to the inner-canthi line.

    Interpolates the inner-canthus segment's z at the nasion's xy projection; positive
    means the crest stands proud of the canthi line (toward the camera) — i.e. a high
    bridge that can carry a frame; low/negative suggests a low-bridge fit.
    """
    a = mm[sides.right.inner_canthus]
    b = mm[sides.left.inner_canthus]
    nasion = mm[idx.NASION]
    d = (b - a)[:2]
    denom = float(np.dot(d, d))
    t = 0.5 if denom == 0.0 else float(np.clip(np.dot((nasion - a)[:2], d) / denom, 0.0, 1.0))
    line_z = float(a[2] + t * (b[2] - a[2]))
    return line_z - float(nasion[2])


def zygoma_width_mm(mm: np.ndarray) -> float:
    """Bizygomatic face width: 3-D distance 234 <-> 454 (symmetric, side-agnostic)."""
    return dist(mm[idx.FACE_SIDE_RIGHT], mm[idx.FACE_SIDE_LEFT])


def temple_width_mm(mm: np.ndarray) -> float:
    """Temple-to-temple width: 3-D distance 127 <-> 356 (symmetric, side-agnostic)."""
    return dist(mm[idx.TEMPLE_RIGHT], mm[idx.TEMPLE_LEFT])


def face_wrap_radius_mm(mm: np.ndarray) -> float:
    """Radius of the horizontal-plane circle through the two face sides and the nasion.

    Fit in the (x, z) plane; a smaller radius means a more curved face (more face-form
    wrap wanted), a large radius a flat face.
    """
    pts = mm[[idx.FACE_SIDE_RIGHT, idx.NASION, idx.FACE_SIDE_LEFT]][:, [0, 2]]
    return fit_circle_radius(pts)


def cheek_clearance_mm(mm: np.ndarray, sides: ResolvedSides) -> PerSide:
    """Vertical gap proxy from the lower orbit (lower lid) down to the malar point.

    Larger values leave more room before a lens bottom touches the cheek."""

    def gap(side: SideIndices) -> float:
        return float(mm[side.mid_cheek, 1] - mm[side.lower_lid, 1])

    return PerSide(right=gap(sides.right), left=gap(sides.left))


def vertex_estimate_mm(mm: np.ndarray, sides: ResolvedSides, has_iris: bool) -> float:
    """Crude vertex-distance estimate: mean pupil z minus nasion z.

    Proxy: a frame front resting on the nose bridge sits near the nasion's depth, so the
    z gap back to the corneal plane approximates vertex distance. Low confidence."""
    right, left = _pupil_centers(mm, sides, has_iris)
    return float((right[2] + left[2]) / 2.0 - mm[idx.NASION, 2])


def canthal_tilt_deg(mm: np.ndarray, sides: ResolvedSides) -> PerSide:
    """Canthal tilt per eye: angle of the outer-minus-inner canthus vector to horizontal.

    Positive = outer canthus HIGHER than inner (upward tilt); the y-down image axis is
    accounted for here."""

    def tilt(side: SideIndices) -> float:
        v = mm[side.outer_canthus] - mm[side.inner_canthus]
        return -angle_to_horizontal_deg(v)

    return PerSide(right=tilt(sides.right), left=tilt(sides.left))


def pupil_height_ratio(mm: np.ndarray, sides: ResolvedSides, has_iris: bool) -> PerSide:
    """Pupil's vertical position within the palpebral aperture (0 = lower lid, 1 = upper).

    A degenerate (closed-eye) aperture returns 0.5."""
    right, left = _pupil_centers(mm, sides, has_iris)

    def ratio(pupil: np.ndarray, side: SideIndices) -> float:
        upper_y = float(mm[side.upper_lid, 1])
        lower_y = float(mm[side.lower_lid, 1])
        aperture = lower_y - upper_y
        if aperture <= 1e-9:
            return 0.5
        return (lower_y - float(pupil[1])) / aperture

    return PerSide(right=ratio(right, sides.right), left=ratio(left, sides.left))


def hinge_to_ear_mm(mm: np.ndarray, sides: ResolvedSides) -> PerSide:
    """ESTIMATE of hinge-to-ear (tragus) length per side.

    The hinge sits roughly in the frame-front plane near the outer canthus; the tragus
    sits ~15-20 mm behind the lateral face-oval point. Estimate = z depth from outer
    canthus to face side + TRAGUS_BEHIND_FACE_OVAL_MM. Total temple length adds
    TEMPLE_BEND_ALLOWANCE_MM on top. Low confidence (mesh z at the face boundary is
    weak and there are no true ear landmarks)."""

    def est(side: SideIndices) -> float:
        depth = float(mm[side.face_side, 2] - mm[side.outer_canthus, 2])
        return depth + TRAGUS_BEHIND_FACE_OVAL_MM

    return PerSide(right=est(sides.right), left=est(sides.left))


def ear_height_asymmetry_mm(mm: np.ndarray, sides: ResolvedSides) -> float:
    """ESTIMATE of ear-height asymmetry from the pre-auricular face-oval proxies.

    Positive = the subject's RIGHT ear sits higher (its proxy has the smaller image y),
    meaning the right temple would need a higher bend. Low confidence."""
    return float(mm[sides.left.face_side, 1] - mm[sides.right.face_side, 1])


def behind_ear_estimate(mm: np.ndarray, sides: ResolvedSides) -> dict[str, BehindEar]:
    """ESTIMATE of the behind-ear temple-bend drop and angle per side.

    Drop = BEHIND_EAR_DROP_FRACTION of the vertical gap from the pre-auricular proxy
    down to the gonion (jaw-angle) landmark — the mastoid the temple tip hugs lies
    between them. Angle = arctan(drop / BEHIND_EAR_BEND_RUN_MM). Low confidence."""

    def est(side: SideIndices) -> BehindEar:
        vertical_gap = float(mm[side.jaw, 1] - mm[side.face_side, 1])
        drop = max(BEHIND_EAR_DROP_FRACTION * vertical_gap, 0.0)
        angle = float(np.degrees(np.arctan2(drop, BEHIND_EAR_BEND_RUN_MM)))
        return BehindEar(drop_mm=drop, angle_deg=angle)

    return {"right": est(sides.right), "left": est(sides.left)}


def compute_measurements(landmarks: LandmarkSet, pd_mm: float) -> FaceMeasurements:
    """Compute every FaceMeasurements field from a landmark set and the entered PD."""
    pts = to_unit_space(landmarks)
    scale, scale_warnings = mm_per_unit(pd_mm, pts, landmarks.has_iris)
    mm = pts * scale
    sides = resolve_sides(mm)
    has_iris = landmarks.has_iris

    quality = MeasurementQuality(
        low_confidence_fields=list(LOW_CONFIDENCE_FIELDS),
        warnings=list(scale_warnings),
    )
    zygoma = zygoma_width_mm(mm)
    low, high = ZYGOMA_PLAUSIBLE_MM
    if not low <= zygoma <= high:
        quality.scale_suspect = True
        quality.warnings.append(f"scale_suspect_zygoma_width_{zygoma:.1f}mm")

    return FaceMeasurements(
        mm_per_unit=scale,
        pd_binocular_mm=binocular_pd_mm(mm, sides, has_iris),
        pd_monocular_mm=monocular_pd_mm(mm, sides, has_iris),
        bridge=bridge_widths_mm(mm),
        bridge_crest_height_mm=bridge_crest_height_mm(mm, sides),
        zygoma_width_mm=zygoma,
        temple_width_mm=temple_width_mm(mm),
        face_wrap_radius_mm=face_wrap_radius_mm(mm),
        cheek_clearance_mm=cheek_clearance_mm(mm, sides),
        hinge_to_ear_mm=hinge_to_ear_mm(mm, sides),
        ear_height_asymmetry_mm=ear_height_asymmetry_mm(mm, sides),
        behind_ear=behind_ear_estimate(mm, sides),
        vertex_estimate_mm=vertex_estimate_mm(mm, sides, has_iris),
        canthal_tilt_deg=canthal_tilt_deg(mm, sides),
        pupil_height_ratio=pupil_height_ratio(mm, sides, has_iris),
        quality=quality,
    )
