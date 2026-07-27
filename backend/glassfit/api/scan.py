"""POST /scan — camera frames in, canonical landmarks + overlay geometry out."""

from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile

from glassfit.api.deps import DetectorDep, RepoDep, SettingsDep
from glassfit.schemas import ScanResponse
from glassfit.services.scan_service import run_scan

router = APIRouter(tags=["scan"])

_SIDE_DESC = (
    "head-turn frames ({} turn). The label is ADVISORY only — the exposed side is "
    "classified geometrically from each frame, so mislabeled bursts still work."
)


@router.post("/scan", response_model=ScanResponse)
def scan(
    request: Request,
    detector: DetectorDep,
    repo: RepoDep,
    settings: SettingsDep,
    frames: Annotated[list[UploadFile], File(description="frontal JPEG/PNG camera frames")],
    frames_left: Annotated[
        list[UploadFile] | None, File(description=_SIDE_DESC.format("left"))
    ] = None,
    frames_right: Annotated[
        list[UploadFile] | None, File(description=_SIDE_DESC.format("right"))
    ] = None,
) -> ScanResponse:
    # sync def: FastAPI runs this on its threadpool (mediapipe inference is blocking)
    blobs = [f.file.read() for f in frames]
    side_blobs = [f.file.read() for f in (frames_left or []) + (frames_right or [])]
    return run_scan(
        blobs,
        detector=detector,
        repo=repo,
        settings=settings,
        side_blobs=side_blobs,
        lock=request.app.state.detector_lock,
    )
