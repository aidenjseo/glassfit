"""Scan endpoint schemas."""

from pydantic import BaseModel

from .common import ImageSize, LandmarkSet, Point2


class FrameReport(BaseModel):
    index: int
    accepted: bool
    reject_reason: str | None = None  # "no_face" | "multiple_faces" | "yaw_exceeded" | ...
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None


class ScanQuality(BaseModel):
    frames_received: int
    frames_used: int
    frame_reports: list[FrameReport]
    landmark_dispersion_px: float
    ok: bool
    warnings: list[str] = []


class OverlaySegment(BaseModel):
    """A named measurement line for the frontend to draw over the captured frame (pixels)."""

    name: str  # "pd" | "bridge_crest" | "zygoma_width" | "temple_width" | ...
    start: Point2
    end: Point2
    label: str


class ScanResponse(BaseModel):
    scan_id: str
    landmarks: LandmarkSet  # normalized, canonical (median-aggregated) set
    landmarks_px: list[Point2]  # same points in pixels of the best frame, for overlay drawing
    key_points: dict[str, Point2]  # named anchors (pupils, nasion, zygoma, ...) in pixels
    overlay_segments: list[OverlaySegment]
    image_size: ImageSize
    best_frame_index: int
    quality: ScanQuality
