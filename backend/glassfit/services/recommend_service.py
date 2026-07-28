"""Recommendation orchestration: request -> measurements -> rules -> persistence."""

from glassfit.config import Settings
from glassfit.errors import InvalidLandmarks, NotFound
from glassfit.measure.extractor import compute_measurements
from glassfit.rules.engine import recommend
from glassfit.rules.params import load_rule_params
from glassfit.schemas import (
    FaceMeasurements,
    MeasurementRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from glassfit.schemas.common import LandmarkSet
from glassfit.storage.repo import Repo


def _apply_side_view_refinement(measurements: FaceMeasurements, scan: dict) -> FaceMeasurements:
    """Override the crude frontal hinge-to-ear estimate with side-view values."""
    side = scan["quality"].side_views  # repo always returns a parsed ScanQuality
    if side is None:
        return measurements
    refined = side.hinge_to_ear_mm
    if refined.get("right") is None and refined.get("left") is None:
        return measurements
    current = measurements.hinge_to_ear_mm
    new_side = current.model_copy(
        update={
            "right": refined.get("right") or current.right,
            "left": refined.get("left") or current.left,
        }
    )
    new_quality = measurements.quality.model_copy(
        update={
            "warnings": [*measurements.quality.warnings, "hinge_to_ear_refined_from_side_views"]
        }
    )
    return measurements.model_copy(update={"hinge_to_ear_mm": new_side, "quality": new_quality})


def _compute_or_422(landmarks: LandmarkSet, pd_mm: float) -> FaceMeasurements:
    """Degenerate geometry raises ValueError deep in numpy — surface it as a 422."""
    try:
        return compute_measurements(landmarks, pd_mm)
    except ValueError as exc:
        raise InvalidLandmarks(f"landmarks do not form a measurable face: {exc}") from exc


def _resolve_measurements(
    req: RecommendationRequest | MeasurementRequest, repo: Repo
) -> tuple[FaceMeasurements, str | None]:
    """Resolve the exactly-one input source into measurements (+ originating scan_id)."""
    if req.scan_id is not None:
        scan = repo.get_scan(req.scan_id)
        if scan is None:
            raise NotFound(f"scan {req.scan_id!r} not found", details={"scan_id": req.scan_id})
        measurements = _compute_or_422(scan["landmarks"], req.pd_mm)
        return _apply_side_view_refinement(measurements, scan), req.scan_id
    if req.landmarks is not None:
        return _compute_or_422(req.landmarks, req.pd_mm), None
    assert isinstance(req, RecommendationRequest) and req.measurements is not None
    return req.measurements, None


def run_measurements(req: MeasurementRequest, *, repo: Repo) -> FaceMeasurements:
    measurements, _ = _resolve_measurements(req, repo)
    return measurements


def run_recommendation(
    req: RecommendationRequest, *, repo: Repo, settings: Settings
) -> RecommendationResponse:
    measurements, scan_id = _resolve_measurements(req, repo)
    params = load_rule_params(settings.rules_path)
    rec = recommend(measurements, req.rx, req.lens_intent, params, pd_monocular=req.pd_monocular)
    request_extras = {
        "rx": req.rx.model_dump(mode="json") if req.rx else None,
        "lens_intent": req.lens_intent,
        "pd_monocular": req.pd_monocular.model_dump(mode="json") if req.pd_monocular else None,
    }
    repo.save_recommendation(
        rec,
        scan_id=scan_id,
        user_id=req.user_id,
        pd_mm=req.pd_mm,
        mm_per_unit=measurements.mm_per_unit,
        measurements=measurements,
        request_extras=request_extras,
    )
    return RecommendationResponse(**rec.model_dump(), measurements=measurements, scan_id=scan_id)
