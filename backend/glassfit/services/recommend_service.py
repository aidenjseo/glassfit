"""Recommendation orchestration: request -> measurements -> rules -> persistence."""

from glassfit.config import Settings
from glassfit.errors import NotFound
from glassfit.measure.extractor import compute_measurements
from glassfit.rules.engine import recommend
from glassfit.rules.params import load_rule_params
from glassfit.schemas import (
    FaceMeasurements,
    MeasurementRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from glassfit.storage.repo import Repo


def _resolve_measurements(
    req: RecommendationRequest | MeasurementRequest, repo: Repo
) -> tuple[FaceMeasurements, str | None]:
    """Resolve the exactly-one input source into measurements (+ originating scan_id)."""
    if req.scan_id is not None:
        scan = repo.get_scan(req.scan_id)
        if scan is None:
            raise NotFound(f"scan {req.scan_id!r} not found", details={"scan_id": req.scan_id})
        return compute_measurements(scan["landmarks"], req.pd_mm), req.scan_id
    if req.landmarks is not None:
        return compute_measurements(req.landmarks, req.pd_mm), None
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
