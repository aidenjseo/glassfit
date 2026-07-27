"""Phase 2: rank catalog frames against recommended fit dimensions.

Two stages, mirroring how an optician narrows a wall of frames:

1. HARD FILTERS — dealbreakers, not preferences: a bridge more than ±2 mm off sits
   visibly wrong on the nose; a temple meaningfully shorter than required cannot
   reach behind the ear.
2. WEIGHTED SOFT SCORE — remaining frames ranked by a weighted, tolerance-normalized
   L1 distance over the five dimensions. DBL dominates (the nose carries the frame),
   then A (overall width), then temple/B/ED. A low-bridge preference (from a low
   bridge-crest measurement) gives pad-adjustable / low-bridge frames a bonus.

Pure functions; all tunables in MatchParams so a later phase can externalize them.
"""

from dataclasses import dataclass, field

from glassfit.schemas import CatalogFrame, FrameDims, FrameMatch

_DIMS = ("a_mm", "b_mm", "dbl_mm", "ed_mm", "temple_length_mm")
_FRAME_ATTR = {**{d: d for d in _DIMS}, "temple_length_mm": "temple_mm"}


@dataclass(frozen=True)
class MatchParams:
    dbl_hard_tolerance_mm: float = 2.0
    temple_short_slack_mm: float = 2.0  # frame temple may be at most this much shorter
    # weight and full-penalty span (mm) per dimension: penalty = weight * min(1, |d|/span)
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "dbl_mm": 0.35,
            "a_mm": 0.30,
            "temple_length_mm": 0.15,
            "b_mm": 0.10,
            "ed_mm": 0.10,
        }
    )
    spans_mm: dict[str, float] = field(
        default_factory=lambda: {
            "dbl_mm": 2.0,
            "a_mm": 4.0,
            "temple_length_mm": 6.0,
            "b_mm": 6.0,
            "ed_mm": 6.0,
        }
    )
    low_bridge_bonus: float = 0.08  # penalty reduction for pad-adjustable/low-bridge frames


DEFAULT_PARAMS = MatchParams()


def _deltas(frame: CatalogFrame, targets: FrameDims) -> dict[str, float]:
    """frame minus target, per dimension (positive = frame is larger)."""
    return {
        dim: round(getattr(frame, _FRAME_ATTR[dim]) - getattr(targets, dim), 2) for dim in _DIMS
    }


def _passes_hard_filters(deltas: dict[str, float], params: MatchParams) -> bool:
    return (
        abs(deltas["dbl_mm"]) <= params.dbl_hard_tolerance_mm
        and deltas["temple_length_mm"] >= -params.temple_short_slack_mm
    )


def _flags(frame: CatalogFrame, deltas: dict[str, float], prefer_low_bridge: bool) -> list[str]:
    flags: list[str] = []
    dbl = deltas["dbl_mm"]
    if dbl <= -1.0:
        flags.append(f"Bridge {abs(dbl):.0f} mm narrow — will sit slightly high on the nose")
    elif dbl >= 1.0:
        flags.append(f"Bridge {dbl:.0f} mm wide — may sit low; pads can compensate")
    if deltas["a_mm"] >= 3.0:
        flags.append("Runs wide for your face")
    elif deltas["a_mm"] <= -3.0:
        flags.append("Runs narrow for your face")
    if deltas["temple_length_mm"] < 0:
        flags.append("Temple slightly short — expect a tighter bend")
    accommodates_low_bridge = frame.low_bridge_fit or frame.nose_pads == "adjustable"
    if prefer_low_bridge and accommodates_low_bridge:
        flags.append("Low-bridge friendly ✓")
    elif prefer_low_bridge and not accommodates_low_bridge:
        flags.append("Fixed pads on a low bridge crest — may slide")
    if frame.spring_hinge:
        flags.append("Spring hinges")
    return flags


def score_frame(
    frame: CatalogFrame,
    targets: FrameDims,
    *,
    prefer_low_bridge: bool = False,
    params: MatchParams = DEFAULT_PARAMS,
) -> FrameMatch | None:
    """Score one frame against the targets; None when a hard filter rejects it."""
    deltas = _deltas(frame, targets)
    if not _passes_hard_filters(deltas, params):
        return None
    penalty = sum(
        params.weights[dim] * min(1.0, abs(deltas[dim]) / params.spans_mm[dim]) for dim in _DIMS
    )
    if prefer_low_bridge and (frame.low_bridge_fit or frame.nose_pads == "adjustable"):
        # Applied as a penalty REDUCTION (not an additive bonus): the score can never
        # saturate at 1.0, so ranking among low-bridge frames stays strictly ordered
        # by dimensional fit and only a true perfect match displays as 100%.
        penalty *= 1.0 - params.low_bridge_bonus
    return FrameMatch(
        frame=frame,
        fit_score=round(max(0.0, 1.0 - penalty), 3),
        deltas=deltas,
        flags=_flags(frame, deltas, prefer_low_bridge),
    )


def match_frames(
    frames: list[CatalogFrame],
    targets: FrameDims,
    *,
    prefer_low_bridge: bool = False,
    limit: int = 8,
    params: MatchParams = DEFAULT_PARAMS,
) -> list[FrameMatch]:
    """Rank the catalog against the targets: hard-filter, score, best first."""
    matches = [
        m
        for f in frames
        if (m := score_frame(f, targets, prefer_low_bridge=prefer_low_bridge, params=params))
        is not None
    ]
    matches.sort(key=lambda m: (-m.fit_score, m.frame.frame_id))
    return matches[:limit]
