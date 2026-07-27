"""Fit-feedback schemas — the labels for the future ML residual model."""

from pydantic import BaseModel, Field

from .recommendation import FrameDims


class AdjustmentsMade(BaseModel):
    """Deltas relative to the recommendation — the residual regression targets."""

    panto_delta_deg: float | None = None
    face_form_delta_deg: float | None = None
    vertex_delta_mm: float | None = None
    temple_bend_delta_mm: float | None = None
    temple_tip_angle_delta_deg: float | None = None
    left_raise_delta_mm: float | None = None
    nose_pad_splay_delta_deg: float | None = None
    nose_pad_spread_delta_mm: float | None = None
    oc_height_delta_mm: float | None = None


class FeedbackIn(BaseModel):
    recommendation_id: str
    frame_id: str | None = None  # catalog frame worn, if any
    frame_worn: FrameDims | None = None  # else free-form dims of the frame tried
    nose_pressure: int = Field(ge=1, le=5)
    temple_pressure: int = Field(ge=1, le=5)
    slips: bool
    cheek_touch: bool
    adjustments: AdjustmentsMade | None = None
    comments: str | None = None
    user_id: str = "local"


class FeedbackOut(BaseModel):
    feedback_id: str
