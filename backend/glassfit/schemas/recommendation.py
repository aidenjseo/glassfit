"""Recommendation schemas — the output maps 1:1 to the project spec's JSON I/O sketch."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .common import LandmarkSet, PerSide
from .measurements import FaceMeasurements, Rx

LensIntent = Literal[
    "single_vision_distance",
    "single_vision_near",
    "progressive",
    "office",
    "sun",
]


class RecommendationRequest(BaseModel):
    pd_mm: float = Field(ge=45, le=80)
    scan_id: str | None = None
    landmarks: LandmarkSet | None = None
    measurements: FaceMeasurements | None = None
    pd_monocular: PerSide | None = None  # user-provided monocular PDs (right=OD, left=OS)
    rx: Rx | None = None
    lens_intent: LensIntent | None = None
    user_id: str = "local"

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "RecommendationRequest":
        provided = [x is not None for x in (self.scan_id, self.landmarks, self.measurements)]
        if sum(provided) != 1:
            raise ValueError("provide exactly one of scan_id, landmarks, or measurements")
        # bounds on the request's monocular PDs only — the shared PerSide type also
        # carries signed/zero outputs (inset, raise) where these bounds would be wrong
        if self.pd_monocular is not None:
            for side in (self.pd_monocular.right, self.pd_monocular.left):
                if not 20.0 <= side <= 45.0:
                    raise ValueError("each monocular PD must be between 20 and 45 mm")
        return self


class FrameDims(BaseModel):
    a_mm: float  # lens width
    b_mm: float  # lens height
    dbl_mm: float  # distance between lenses (bridge)
    ed_mm: float  # effective diameter
    temple_length_mm: float


class AsWorn(BaseModel):
    pantoscopic_deg: float  # spec rule: 5-8 deg
    face_form_deg: float  # wrap, 5-10 deg
    vertex_mm: float  # 12-14 mm


class Optics(BaseModel):
    pd_monocular_mm: PerSide
    oc_height_mm: PerSide
    inset_mm: PerSide


class NosePads(BaseModel):
    size: Literal["S", "M", "L"]
    splay_deg: float
    flare_deg: float
    drop_mm: float


class Temples(BaseModel):
    bend_point_mm_from_hinge: PerSide
    tip_angle_deg: float
    raise_mm: PerSide  # per-temple raise to level the frame (0 = no change)


class Comfort(BaseModel):
    predicted_slip: float = Field(ge=0, le=1)
    nose_pressure: float = Field(ge=0, le=1)
    temple_pressure: float = Field(ge=0, le=1)


class RuleTrace(BaseModel):
    rule_id: str
    inputs: dict[str, float]
    raw_value: float
    clamped: bool


class Recommendation(BaseModel):
    recommendation_id: str
    ruleset_version: str
    frame: FrameDims
    as_worn: AsWorn
    optics: Optics
    nose_pads: NosePads
    temples: Temples
    comfort: Comfort
    notes: list[str]
    rule_trace: dict[str, RuleTrace]  # keyed by output field path, for ML residual learning


class RecommendationResponse(Recommendation):
    """API response: the recommendation plus the measurements it was derived from."""

    measurements: FaceMeasurements
    scan_id: str | None = None
