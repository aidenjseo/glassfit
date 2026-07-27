"""POST /recommendations — the full fit recommendation (measurements included)."""

from fastapi import APIRouter

from glassfit.api.deps import RepoDep, SettingsDep
from glassfit.schemas import RecommendationRequest, RecommendationResponse
from glassfit.services.recommend_service import run_recommendation

router = APIRouter(tags=["recommendations"])


@router.post("/recommendations", response_model=RecommendationResponse)
def recommendations(
    req: RecommendationRequest, repo: RepoDep, settings: SettingsDep
) -> RecommendationResponse:
    return run_recommendation(req, repo=repo, settings=settings)
