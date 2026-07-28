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


class SideViewSummary(BaseModel):
    """What the head-turn bursts contributed.

    Sides are SUBJECT-anatomical: turning the head to the subject's LEFT exposes the
    subject's RIGHT cheek/ear to the camera (counted under ``right_frames``).
    """

    right_frames: int = 0  # usable frames exposing the subject-right side
    left_frames: int = 0
    hinge_to_ear_mm: dict[str, float | None] = {"right": None, "left": None}


class ScanQuality(BaseModel):
    frames_received: int
    frames_used: int
    frame_reports: list[FrameReport]
    landmark_dispersion_px: float
    ok: bool
    warnings: list[str] = []
    side_views: SideViewSummary | None = None


class OverlaySegment(BaseModel):
    """A named measurement line for the frontend to draw over the captured frame (pixels)."""

    name: str  # "pd" | "bridge_crest" | "zygoma_width" | "temple_width" | ...
    start: Point2
    end: Point2
    label: str


class ProbeResponse(BaseModel):
    """Lightweight live-preview feedback: is the user framed well for a scan?"""

    face_count: int
    face_width_frac: float | None = None  # face width / image width
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    guidance: str  # "no_face" | "closer" | "back" | "ok"
    message: str  # human copy for the on-stage hint


class ScanResponse(BaseModel):
    scan_id: str
    # Normalized canonical (median-aggregated) set, aligned to the best frame's geometry.
    # Pixel positions are derivable: (x * image_size.width, y * image_size.height).
    landmarks: LandmarkSet
    key_points: dict[str, Point2]  # named anchors (pupils, nasion, zygoma, ...) in pixels
    overlay_segments: list[OverlaySegment]
    image_size: ImageSize
    best_frame_index: int
    quality: ScanQuality
