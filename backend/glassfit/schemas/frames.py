"""Frame catalog schemas (Phase 2 matching builds on these)."""

from typing import Literal

from pydantic import BaseModel


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
