"""GET /frames — the sample frame catalog (Phase 2 adds ranked matching)."""

from fastapi import APIRouter

from glassfit.api.deps import RepoDep
from glassfit.schemas import FrameListResponse

router = APIRouter(tags=["frames"])


@router.get("/frames", response_model=FrameListResponse)
def list_frames(
    repo: RepoDep,
    a_min: float | None = None,
    a_max: float | None = None,
    dbl_min: float | None = None,
    dbl_max: float | None = None,
    temple: float | None = None,
    limit: int = 50,
) -> FrameListResponse:
    return FrameListResponse(
        frames=repo.list_frames(
            a_min=a_min, a_max=a_max, dbl_min=dbl_min, dbl_max=dbl_max, temple=temple, limit=limit
        )
    )
