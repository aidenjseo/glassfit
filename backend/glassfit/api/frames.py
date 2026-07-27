"""Frame catalog endpoints: listing + Phase-2 ranked fit matching."""

from fastapi import APIRouter

from glassfit.api.deps import RepoDep, SettingsDep
from glassfit.catalog.match import match_frames
from glassfit.errors import NotFound
from glassfit.rules.params import load_rule_params
from glassfit.schemas import FrameListResponse, FrameMatchRequest, FrameMatchResponse

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


@router.post("/frames/match", response_model=FrameMatchResponse)
def frames_match(
    req: FrameMatchRequest, repo: RepoDep, settings: SettingsDep
) -> FrameMatchResponse:
    prefer_low_bridge = False
    if req.recommendation_id is not None:
        stored = repo.get_recommendation(req.recommendation_id)
        if stored is None:
            raise NotFound(
                f"recommendation {req.recommendation_id!r} not found",
                details={"recommendation_id": req.recommendation_id},
            )
        targets = stored["recommendation"].frame
        crest = stored["measurements"].bridge_crest_height_mm
        low_crest = load_rule_params(settings.rules_path).nose_pads.low_crest_adjustable_mm
        prefer_low_bridge = crest < low_crest
    else:
        assert req.targets is not None  # guaranteed by the request validator
        targets = req.targets
    matches = match_frames(
        repo.list_frames(limit=1000),
        targets,
        prefer_low_bridge=prefer_low_bridge,
        limit=req.limit,
    )
    return FrameMatchResponse(targets=targets, matches=matches)
