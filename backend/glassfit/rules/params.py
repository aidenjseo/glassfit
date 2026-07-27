"""Typed rule parameters — a 1:1 pydantic mirror of rules/defaults.yaml.

Units: lengths in millimeters, angles in degrees. `extra="forbid"` on every model turns a
typo in the yaml into a load-time error instead of a silently ignored knob.
"""

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "defaults.yaml"


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- as_worn ---------------------------------------------------------------


class PantoscopicParams(_Params):
    default_deg: float
    min_deg: float
    max_deg: float
    canthal_tilt_gain: float


class FaceFormParams(_Params):
    base_deg: float
    min_deg: float
    max_deg: float
    wrap_radius_ref_mm: float
    gain_deg_per_mm: float


class VertexParams(_Params):
    default_mm: float
    min_mm: float
    max_mm: float
    measured_gain: float


class AsWornParams(_Params):
    pantoscopic: PantoscopicParams
    face_form: FaceFormParams
    vertex: VertexParams


# --- rx guardrails ---------------------------------------------------------


class RxGuardrails(_Params):
    high_plus_sphere_d: float
    high_plus_panto_reduction_deg: float
    high_minus_sphere_d: float
    high_minus_vertex_target_mm: float
    strong_rx_abs_sphere_d: float
    strong_rx_wrap_cap_deg: float


# --- frame -----------------------------------------------------------------


class AFromZygomaParams(_Params):
    slope: float
    intercept_mm: float
    margin_mm: float
    min_mm: float
    max_mm: float


class DblParams(_Params):
    offset_fixed_pads_mm: float
    offset_adjustable_pads_mm: float
    min_mm: float
    max_mm: float


class FramePdParams(_Params):
    decentration_warn_mm: float


class BFromPupilParams(_Params):
    base_b_over_a: float
    pupil_ratio_gain_mm: float
    progressive_extra_b_mm: float
    min_b_progressive_mm: float
    min_b_mm: float
    max_b_mm: float
    pupil_target_frac: dict[str, float]

    @field_validator("pupil_target_frac")
    @classmethod
    def _has_default(cls, v: dict[str, float]) -> dict[str, float]:
        if "default" not in v:
            raise ValueError("pupil_target_frac must contain a 'default' key")
        return v


class FrameParams(_Params):
    a_from_zygoma: AFromZygomaParams
    dbl: DblParams
    frame_pd: FramePdParams
    b_from_pupil: BFromPupilParams
    ed_margin_mm: float


# --- temples ---------------------------------------------------------------


class TempleParams(_Params):
    standard_lengths_mm: list[float] = Field(min_length=1)
    bend_allowance_mm: float
    bend_point_offset_mm: float
    tip_angle_default_deg: float
    tip_angle_min_deg: float
    tip_angle_max_deg: float
    raise_threshold_mm: float
    raise_max_mm: float


# --- nose pads -------------------------------------------------------------


class PadSizeThresholds(_Params):
    low_crest_mm: float
    high_crest_mm: float


class SplayParams(_Params):
    default_deg: float
    min_deg: float
    max_deg: float
    bridge_width_ref_mm: float
    gain_deg_per_mm: float


class DropParams(_Params):
    default_mm: float
    min_mm: float
    max_mm: float
    crest_ref_mm: float
    gain_mm_per_mm: float


class NosePadParams(_Params):
    size_thresholds: PadSizeThresholds
    low_crest_adjustable_mm: float
    splay: SplayParams
    flare_default_deg: float
    drop: DropParams


# --- optics ----------------------------------------------------------------


class OpticsParams(_Params):
    inset_per_diopter_mm: float
    near_inset_max_mm: float
    oc_ratio_gain_mm: float
    near_inset_mm: dict[str, float]

    @field_validator("near_inset_mm")
    @classmethod
    def _has_default(cls, v: dict[str, float]) -> dict[str, float]:
        if "default" not in v:
            raise ValueError("near_inset_mm must contain a 'default' key")
        return v


# --- comfort ---------------------------------------------------------------


class TemplePressureParams(_Params):
    zero_at_mm: float
    one_at_mm: float


class NosePressureParams(_Params):
    base: float
    crest_ref_mm: float
    crest_range_mm: float


class SlipParams(_Params):
    crest_weight: float
    lens_area_weight: float
    crest_ref_mm: float
    crest_range_mm: float
    lens_area_ref_mm2: float
    lens_area_range_mm2: float


class ComfortParams(_Params):
    endpiece_mm: float
    spring_hinge_note_mm: float
    temple_pressure: TemplePressureParams
    nose_pressure: NosePressureParams
    slip: SlipParams


# --- rounding --------------------------------------------------------------


class RoundingParams(_Params):
    length_step_mm: float
    angle_step_deg: float
    inset_step_mm: float
    comfort_decimals: int


# --- root ------------------------------------------------------------------


class RuleParams(_Params):
    ruleset_version: str
    as_worn: AsWornParams
    rx_guardrails: RxGuardrails
    frame: FrameParams
    temples: TempleParams
    nose_pads: NosePadParams
    optics: OpticsParams
    comfort: ComfortParams
    rounding: RoundingParams


@cache
def load_rule_params(path: Path = DEFAULT_RULES_PATH) -> RuleParams:
    """Load and validate a ruleset yaml. Cached per path (rulesets are immutable at runtime)."""
    with path.open("rb") as f:
        data = yaml.safe_load(f)
    return RuleParams.model_validate(data)
