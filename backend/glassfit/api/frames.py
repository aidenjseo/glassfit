"""Frame catalog endpoints: listing, Phase-2 ranked fit matching, match ratings."""

from fastapi import APIRouter

from glassfit.api.deps import RepoDep, SettingsDep
from glassfit.catalog.match import MatchContext, context_for_recommendation, match_frames
from glassfit.errors import NotFound
from glassfit.rules.params import load_rule_params
from glassfit.schemas import (
    FrameListResponse,
    FrameMatchRequest,
    FrameMatchResponse,
    MatchRatingIn,
    MatchRatingOut,
)

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
    if req.recommendation_id is not None:
        stored = repo.get_recommendation(req.recommendation_id)
        if stored is None:
            raise NotFound(
                f"recommendation {req.recommendation_id!r} not found",
                details={"recommendation_id": req.recommendation_id},
            )
        low_crest = load_rule_params(settings.rules_path).nose_pads.low_crest_adjustable_mm
        ctx = context_for_recommendation(
            stored["recommendation"], stored["measurements"], low_crest
        )
    else:
        assert req.targets is not None  # guaranteed by the request validator
        ctx = MatchContext(targets=req.targets)
    matches = match_frames(repo.list_frames(limit=1000), ctx, limit=req.limit)
    return FrameMatchResponse(targets=ctx.targets, matches=matches)


@router.post("/frames/ratings", response_model=MatchRatingOut, status_code=201)
def rate_match(rating: MatchRatingIn, repo: RepoDep) -> MatchRatingOut:
    return MatchRatingOut(rating_id=repo.save_match_rating(rating))
