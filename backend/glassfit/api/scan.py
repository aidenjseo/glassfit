"""POST /scan — camera frames in, canonical landmarks + overlay geometry out."""

from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile

from glassfit.api.deps import DetectorDep, RepoDep, SettingsDep
from glassfit.schemas import ScanResponse
from glassfit.services.scan_service import run_scan

router = APIRouter(tags=["scan"])


@router.post("/scan", response_model=ScanResponse)
def scan(
    request: Request,
    detector: DetectorDep,
    repo: RepoDep,
    settings: SettingsDep,
    frames: Annotated[list[UploadFile], File(description="1-15 JPEG/PNG camera frames")],
    meta: Annotated[str | None, Form(description="optional capture metadata JSON")] = None,
) -> ScanResponse:
    # sync def: FastAPI runs this on its threadpool (mediapipe inference is blocking)
    blobs = [f.file.read() for f in frames]
    return run_scan(
        blobs,
        detector=detector,
        repo=repo,
        settings=settings,
        lock=request.app.state.detector_lock,
    )
