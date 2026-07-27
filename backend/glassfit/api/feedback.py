"""POST /feedback — log fit feedback (the future ML training labels)."""

from fastapi import APIRouter

from glassfit.api.deps import RepoDep
from glassfit.schemas import FeedbackIn, FeedbackOut

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackOut, status_code=201)
def feedback(fb: FeedbackIn, repo: RepoDep) -> FeedbackOut:
    return FeedbackOut(feedback_id=repo.save_feedback(fb))
