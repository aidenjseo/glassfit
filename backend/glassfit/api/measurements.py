"""POST /measurements — landmarks + PD in, named mm measurements out."""

from fastapi import APIRouter

from glassfit.api.deps import RepoDep
from glassfit.schemas import FaceMeasurements, MeasurementRequest
from glassfit.services.recommend_service import run_measurements

router = APIRouter(tags=["measurements"])


@router.post("/measurements", response_model=FaceMeasurements)
def measurements(req: MeasurementRequest, repo: RepoDep) -> FaceMeasurements:
    return run_measurements(req, repo=repo)
