"""Frame catalog schemas: the catalog itself + Phase-2 fit matching."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .recommendation import FrameDims


class CatalogFrame(BaseModel):
    frame_id: str
    name: str
    shape: str  # rectangle | panto | square | aviator | cat-eye | browline | round | ...
    material: str  # acetate | metal | titanium | TR-90 | nylon | combo
    rim: str  # full | semi | rimless
    a_mm: float
    b_mm: float
    dbl_mm: float
    ed_mm: float
    temple_mm: float
    weight_g: float
    bridge_style: str
    nose_pads: Literal["fixed_acetate", "molded", "adjustable"]
    low_bridge_fit: bool = False
    spring_hinge: bool = False
    default_wrap_deg: float | None = None
    tags: list[str] = []


class FrameListResponse(BaseModel):
    frames: list[CatalogFrame]


class FrameMatchRequest(BaseModel):
    """Rank catalog frames against fit targets.

    Provide EITHER a stored recommendation (preferred — its measurements also drive
    the low-bridge preference) OR inline target dimensions.
    """

    recommendation_id: str | None = None
    targets: FrameDims | None = None
    limit: int = Field(8, ge=1, le=50)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "FrameMatchRequest":
        if (self.recommendation_id is None) == (self.targets is None):
            raise ValueError("provide exactly one of recommendation_id or targets")
        return self


class FrameMatch(BaseModel):
    frame: CatalogFrame
    fit_score: float = Field(ge=0.0, le=1.0)  # 1.0 = perfect dimensional match
    deltas: dict[str, float]  # frame minus target, mm, keyed by FrameDims field
    flags: list[str] = []  # human-readable fit caveats/positives


class FrameMatchResponse(BaseModel):
    targets: FrameDims
    matches: list[FrameMatch]  # best first; hard-filtered frames are absent
