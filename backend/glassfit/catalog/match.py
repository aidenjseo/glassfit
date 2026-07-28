"""Phase 2: rank catalog frames against a face — component-based fit scoring.

The score is a weighted sum of NAMED 0..1 components, each modeling one thing an
optician actually weighs. Components are returned in the API response (explainable
UI) and persisted with match ratings (features for the Phase-3 learning loop):

- optical_centration  frame PD (A + DBL) vs the wearer's PD — the optically
                      dominant criterion: decentration causes prism and edge
                      thickness. The SUM matters, not A and DBL separately.
- bridge_fit          DBL vs the bridge target, ASYMMETRIC (too narrow pinches and
                      rides high — worse than slightly wide, which pads can fix);
                      adjustable pads soften the error and rescue low bridges.
- width_fit           frame front vs measured temple width: pinch when the head is
                      wider, slide/oversized when the frame is much wider.
- lens_height         B vs target, plus a cheek-contact check: lens depth below the
                      pupil must clear the measured cheek clearance.
- temple_fit          reach is hard-filtered; excess length is a mild penalty
                      (bend sits far behind the ear).
- wrap_harmony        frame wrap vs the recommended as-worn face-form angle.
- weight_slip         heavy frames on a low bridge crest slide.
- shape_affinity      face-proportion heuristics (face length/width, jaw/cheek
                      ratio, canthal tilt) vs frame shape and lens depth — the
                      "optimal frame shapes" of the original spec.

Without measurements (inline dimension targets only) the measurement-dependent
components fall back to target-based or neutral values, so the endpoint still
works — just with less signal. Pure functions; tunables live in MatchParams.
"""

import math
from dataclasses import dataclass, field

from glassfit.schemas import CatalogFrame, FaceMeasurements, FrameDims, FrameMatch

_DIMS = ("a_mm", "b_mm", "dbl_mm", "ed_mm", "temple_length_mm")
_FRAME_ATTR = {**{d: d for d in _DIMS}, "temple_length_mm": "temple_mm"}

NEUTRAL = 0.7  # component value when the data to judge it is absent

# Empirical face-proportion MEANS from 77 real portrait scans — the single live home
# for these ratios (fixtures and test builders import them; migration v4 keeps its own
# frozen literals by design, since shipped backfills must not drift with recalibration).
EMPIRICAL_LENGTH_RATIO = 1.20  # face_length / zygoma
EMPIRICAL_JAW_RATIO = 0.91  # jaw_width / zygoma


def frame_front_width_mm(a_mm: float, dbl_mm: float, endpiece_total_mm: float) -> float:
    """Total frame-front width: both lenses + bridge + end pieces.

    The one home for the ``2A + DBL + endpieces`` formula — the rules engine's
    comfort proxy, the width_fit component, and the synthetic-label generator all
    consume it, so a retuned endpiece allowance can never silently diverge.
    """
    return 2.0 * a_mm + dbl_mm + endpiece_total_mm


# frame shapes considered "angular" vs "rounded" for shape affinity
_ANGULAR_SHAPES = {"rectangle", "square", "geometric", "wayfarer", "browline"}
_ROUNDED_SHAPES = {"round", "oval", "panto", "aviator"}
_UPSWEPT_SHAPES = {"cat-eye", "browline"}


@dataclass(frozen=True)
class MatchParams:
    # hard filters
    dbl_hard_tolerance_mm: float = 2.0
    dbl_hard_tolerance_adjustable_mm: float = 3.0  # pads buy extra room
    temple_short_slack_mm: float = 2.0
    # component weights (must sum to 1.0)
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "optical_centration": 0.20,
            "bridge_fit": 0.22,
            "width_fit": 0.15,
            "lens_height": 0.12,
            "temple_fit": 0.08,
            "wrap_harmony": 0.05,
            "weight_slip": 0.06,
            "shape_affinity": 0.12,
        }
    )
    # full penalty at this much decentration PER LENS ((frame_pd - pd)/2); 1-3mm is
    # routine optical practice, ~4mm+ causes real prism/edge-thickness problems
    centration_span_mm: float = 5.0
    bridge_span_mm: float = 3.5
    bridge_narrow_factor: float = 1.4  # narrow errors count harder than wide ones
    bridge_adjustable_softening: float = 0.6  # pads absorb part of the error
    low_bridge_incompat_factor: float = 0.75
    endpiece_total_mm: float = 12.0
    overhang_pinch_span_mm: float = 8.0  # head wider than frame -> pressure
    overhang_loose_start_mm: float = 8.0  # frame wider than head before penalty
    overhang_loose_span_mm: float = 10.0
    b_span_mm: float = 6.0
    cheek_allowance_mm: float = 4.0
    cheek_excess_span_mm: float = 6.0
    temple_excess_span_mm: float = 12.0
    default_frame_wrap_deg: float = 5.0
    wrap_span_deg: float = 6.0
    weight_norm_g: tuple[float, float] = (8.0, 28.0)
    low_crest_mm: float = 6.0  # matches rules nose_pads.low_crest_adjustable_mm
    # shape affinity — thresholds are EMPIRICAL TERCILES from 77 real portrait scans
    # (length_ratio mean 1.196 sd 0.046; jaw_ratio mean 0.910 sd 0.035). Landmark 10
    # sits near the hairline and 58/288 outside true gonion, so these proxy ratios run
    # higher than textbook anthropometry — calibrate against the proxies, not textbooks.
    long_face_ratio: float = 1.22  # face_length / zygoma above -> "long" (top tercile)
    short_face_ratio: float = 1.175  # below -> "short" (bottom tercile)
    angular_jaw_ratio: float = 0.92  # jaw / zygoma above -> "angular" (top tercile)
    b_over_a_by_length: dict[str, float] = field(
        default_factory=lambda: {"long": 0.80, "balanced": 0.72, "short": 0.66}
    )
    b_over_a_span: float = 0.25
    upswept_canthal_deg: float = -1.0  # mean canthal tilt below -> upswept shapes lift


DEFAULT_PARAMS = MatchParams()


@dataclass(frozen=True)
class MatchContext:
    """Everything the scorer may use; only ``targets`` is required."""

    targets: FrameDims
    measurements: FaceMeasurements | None = None
    face_form_target_deg: float | None = None
    prefer_low_bridge: bool = False


def context_for_recommendation(
    recommendation, measurements: FaceMeasurements, low_crest_mm: float
) -> MatchContext:
    """The canonical stored-recommendation -> MatchContext mapping.

    One home for the prefer-low-bridge rule so the live API and the synthetic-label
    generator can never diverge on it.
    """
    return MatchContext(
        targets=recommendation.frame,
        measurements=measurements,
        face_form_target_deg=recommendation.as_worn.face_form_deg,
        prefer_low_bridge=measurements.bridge_crest_height_mm < low_crest_mm,
    )


def _band01(value: float, span: float) -> float:
    """1.0 at zero error, linearly down to 0.0 at ``span``."""
    return max(0.0, 1.0 - abs(value) / span)


def _band01_tailed(value: float, span: float) -> float:
    """Linear to 0.2 at ``span``, then an exponential tail (continuous, monotone).

    For components where the whole catalog can exceed the tolerance on small/large
    faces (centration, width): a hard floor at 0 would tie every frame and erase
    35% of the ranking signal — with the tail, exceeding the span LESS still wins.
    """
    x = abs(value) / span
    if x <= 1.0:
        return 1.0 - 0.8 * x
    return 0.2 * math.exp(1.0 - x)


def _deltas(frame: CatalogFrame, targets: FrameDims) -> dict[str, float]:
    """frame minus target, per dimension (positive = frame is larger)."""
    return {
        dim: round(getattr(frame, _FRAME_ATTR[dim]) - getattr(targets, dim), 2) for dim in _DIMS
    }


def _accommodates_low_bridge(frame: CatalogFrame) -> bool:
    return frame.low_bridge_fit or frame.nose_pads == "adjustable"


def _passes_hard_filters(
    frame: CatalogFrame, deltas: dict[str, float], params: MatchParams
) -> bool:
    dbl_tol = (
        params.dbl_hard_tolerance_adjustable_mm
        if frame.nose_pads == "adjustable"
        else params.dbl_hard_tolerance_mm
    )
    return (
        abs(deltas["dbl_mm"]) <= dbl_tol
        and deltas["temple_length_mm"] >= -params.temple_short_slack_mm
    )


# --- components (each returns a 0..1 score; flags are collected separately) ----------


def _optical_centration(frame: CatalogFrame, ctx: MatchContext, p: MatchParams) -> float:
    wearer_pd = (
        ctx.measurements.pd_binocular_mm
        if ctx.measurements is not None
        else ctx.targets.a_mm + ctx.targets.dbl_mm
    )
    # Frame PD structurally exceeds wearer PD on almost every face; what matters
    # optically is the DECENTRATION PER LENS the lab must grind in.
    decentration_per_lens = (frame.a_mm + frame.dbl_mm - wearer_pd) / 2.0
    return _band01_tailed(decentration_per_lens, p.centration_span_mm)


def _bridge_fit(
    frame: CatalogFrame, deltas: dict[str, float], ctx: MatchContext, p: MatchParams
) -> float:
    delta = deltas["dbl_mm"]
    effective = delta * (p.bridge_narrow_factor if delta < 0 else 1.0)
    if frame.nose_pads == "adjustable":
        effective *= p.bridge_adjustable_softening
    score = _band01(effective, p.bridge_span_mm)
    if ctx.prefer_low_bridge:
        if _accommodates_low_bridge(frame):
            score = min(1.0, score + 0.1)
        else:
            score *= p.low_bridge_incompat_factor
    return score


def _width_fit(
    frame: CatalogFrame, deltas: dict[str, float], ctx: MatchContext, p: MatchParams
) -> float:
    if ctx.measurements is None:
        return _band01(deltas["a_mm"], 4.0)
    front = frame_front_width_mm(frame.a_mm, frame.dbl_mm, p.endpiece_total_mm)
    overhang = ctx.measurements.temple_width_mm - front  # >0: head wider -> pressure
    if overhang > 0:
        return _band01_tailed(overhang, p.overhang_pinch_span_mm)
    loose = -overhang - p.overhang_loose_start_mm
    return 1.0 if loose <= 0 else _band01_tailed(loose, p.overhang_loose_span_mm)


def _lens_height(
    frame: CatalogFrame, deltas: dict[str, float], ctx: MatchContext, p: MatchParams
) -> float:
    # tailed bands: deep frames on high cheeks can exceed both spans across the whole
    # catalog — a hard floor at 0 would tie them all (same failure class as centration)
    target_term = _band01_tailed(deltas["b_mm"], p.b_span_mm)
    m = ctx.measurements
    if m is None:
        return target_term
    # pupil_height_ratio measures UP from the lens bottom (0 = bottom lid), so the
    # lens depth below the pupil is B * ratio — the OC height itself.
    pupil_frac = (m.pupil_height_ratio.right + m.pupil_height_ratio.left) / 2.0
    below_pupil = frame.b_mm * pupil_frac
    clearance = (m.cheek_clearance_mm.right + m.cheek_clearance_mm.left) / 2.0
    excess = below_pupil - (clearance + p.cheek_allowance_mm)
    cheek_term = 1.0 if excess <= 0 else _band01_tailed(excess, p.cheek_excess_span_mm)
    return 0.4 * target_term + 0.6 * cheek_term


def _temple_fit(deltas: dict[str, float], p: MatchParams) -> float:
    excess = max(0.0, deltas["temple_length_mm"])
    shortage = max(0.0, -deltas["temple_length_mm"])  # within slack, filtered above
    return (
        1.0
        - 0.5 * min(1.0, excess / p.temple_excess_span_mm)
        - 0.5 * min(1.0, shortage / p.temple_short_slack_mm)
    )


def _wrap_harmony(frame: CatalogFrame, ctx: MatchContext, p: MatchParams) -> float:
    if ctx.face_form_target_deg is None:
        return NEUTRAL
    # explicit None check: 0.0 deg is a legitimate declared wrap for a flat front
    frame_wrap = (
        frame.default_wrap_deg if frame.default_wrap_deg is not None else p.default_frame_wrap_deg
    )
    return _band01(frame_wrap - ctx.face_form_target_deg, p.wrap_span_deg)


def _weight_slip(frame: CatalogFrame, ctx: MatchContext, p: MatchParams) -> float:
    lo, hi = p.weight_norm_g
    heaviness = min(1.0, max(0.0, (frame.weight_g - lo) / (hi - lo)))
    if ctx.measurements is None:
        return 1.0 - 0.3 * heaviness
    crest_factor = 1.0 if ctx.measurements.bridge_crest_height_mm < p.low_crest_mm else 0.4
    return 1.0 - 0.5 * heaviness * crest_factor


def _shape_affinity(frame: CatalogFrame, ctx: MatchContext, p: MatchParams) -> float:
    m = ctx.measurements
    if m is None:
        return NEUTRAL
    length_ratio = m.face_length_mm / m.zygoma_width_mm
    face = (
        "long"
        if length_ratio > p.long_face_ratio
        else ("short" if length_ratio < p.short_face_ratio else "balanced")
    )
    # lens-depth proportion: longer faces carry deeper lenses
    preferred = p.b_over_a_by_length[face]
    proportion = _band01(frame.b_mm / frame.a_mm - preferred, p.b_over_a_span)
    # contrast heuristic: angular jaws soften with rounded frames and vice versa
    jaw_ratio = m.jaw_width_mm / m.zygoma_width_mm
    shape = 0.75
    if (
        jaw_ratio >= p.angular_jaw_ratio
        and frame.shape in _ROUNDED_SHAPES
        or jaw_ratio < p.angular_jaw_ratio
        and frame.shape in _ANGULAR_SHAPES
    ):
        shape = 1.0
    mean_canthal = (m.canthal_tilt_deg.right + m.canthal_tilt_deg.left) / 2.0
    if mean_canthal < p.upswept_canthal_deg and frame.shape in _UPSWEPT_SHAPES:
        shape = 1.0  # upswept shapes visually lift low/negative canthal tilt
    return 0.55 * proportion + 0.45 * shape


def _flags(
    frame: CatalogFrame,
    deltas: dict[str, float],
    components: dict[str, float],
    ctx: MatchContext,
) -> list[str]:
    flags: list[str] = []
    dbl = deltas["dbl_mm"]
    if dbl <= -1.0:
        flags.append(f"Bridge {abs(dbl):.0f} mm narrow — will sit slightly high on the nose")
    elif dbl >= 1.0:
        flags.append(f"Bridge {dbl:.0f} mm wide — may sit low; pads can compensate")
    if components["optical_centration"] < 0.6:
        flags.append("Lens centers need noticeable decentration for your PD")
    if components["width_fit"] < 0.6:
        flags.append(
            "Width mismatch for your head"
            if ctx.measurements is not None
            else "Runs wide or narrow for your face"
        )
    if ctx.measurements is not None and components["lens_height"] < 0.7:
        flags.append("Deep lens may sit close to your cheeks")
    if deltas["temple_length_mm"] < 0:
        flags.append("Temple slightly short — expect a tighter bend")
    if ctx.prefer_low_bridge:
        if _accommodates_low_bridge(frame):
            flags.append("Low-bridge friendly ✓")
        else:
            flags.append("Fixed pads on a low bridge crest — may slide")
    if ctx.measurements is not None and components["shape_affinity"] >= 0.85:
        flags.append("Shape suits your face proportions")
    if frame.spring_hinge:
        flags.append("Spring hinges")
    return flags


def score_frame(
    frame: CatalogFrame,
    ctx: MatchContext,
    params: MatchParams = DEFAULT_PARAMS,
) -> FrameMatch | None:
    """Score one frame against the context; None when a hard filter rejects it."""
    deltas = _deltas(frame, ctx.targets)
    if not _passes_hard_filters(frame, deltas, params):
        return None
    components = {
        "optical_centration": _optical_centration(frame, ctx, params),
        "bridge_fit": _bridge_fit(frame, deltas, ctx, params),
        "width_fit": _width_fit(frame, deltas, ctx, params),
        "lens_height": _lens_height(frame, deltas, ctx, params),
        "temple_fit": _temple_fit(deltas, params),
        "wrap_harmony": _wrap_harmony(frame, ctx, params),
        "weight_slip": _weight_slip(frame, ctx, params),
        "shape_affinity": _shape_affinity(frame, ctx, params),
    }
    score = sum(params.weights[name] * value for name, value in components.items())
    return FrameMatch(
        frame=frame,
        fit_score=round(max(0.0, min(1.0, score)), 3),
        components={name: round(value, 3) for name, value in components.items()},
        deltas=deltas,
        flags=_flags(frame, deltas, components, ctx),
    )


def match_frames(
    frames: list[CatalogFrame],
    ctx: MatchContext,
    *,
    limit: int = 8,
    params: MatchParams = DEFAULT_PARAMS,
) -> list[FrameMatch]:
    """Rank the catalog against the context: hard-filter, score, best first."""
    matches = [m for f in frames if (m := score_frame(f, ctx, params)) is not None]
    matches.sort(key=lambda m: (-m.fit_score, m.frame.frame_id))
    return matches[:limit]
