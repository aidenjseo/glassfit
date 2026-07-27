"""Pydantic schemas — the single source of truth for all API and inter-module data shapes."""

from .common import ImageSize, LandmarkSet, PerSide, Point2, Point3
from .feedback import AdjustmentsMade, FeedbackIn, FeedbackOut
from .frames import CatalogFrame, FrameListResponse
from .measurements import (
    BehindEar,
    BridgeWidths,
    FaceMeasurements,
    MeasurementQuality,
    MeasurementRequest,
    Rx,
    RxEye,
)
from .recommendation import (
    AsWorn,
    Comfort,
    FrameDims,
    LensIntent,
    NosePads,
    Optics,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
    RuleTrace,
    Temples,
)
from .scan import FrameReport, OverlaySegment, ScanQuality, ScanResponse, SideViewSummary

__all__ = [
    "AdjustmentsMade",
    "AsWorn",
    "BehindEar",
    "BridgeWidths",
    "CatalogFrame",
    "Comfort",
    "FaceMeasurements",
    "FeedbackIn",
    "FeedbackOut",
    "FrameDims",
    "FrameListResponse",
    "FrameReport",
    "ImageSize",
    "LandmarkSet",
    "LensIntent",
    "MeasurementQuality",
    "MeasurementRequest",
    "NosePads",
    "Optics",
    "OverlaySegment",
    "PerSide",
    "Point2",
    "Point3",
    "Recommendation",
    "RecommendationRequest",
    "RecommendationResponse",
    "RuleTrace",
    "Rx",
    "RxEye",
    "ScanQuality",
    "ScanResponse",
    "SideViewSummary",
    "Temples",
]
