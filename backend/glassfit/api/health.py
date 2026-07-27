"""Liveness + version info."""

from fastapi import APIRouter

from glassfit import __version__
from glassfit.api.deps import SettingsDep, mediapipe_available
from glassfit.rules.params import load_rule_params

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: SettingsDep) -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "mediapipe_available": mediapipe_available(settings),
        "ruleset_version": load_rule_params(settings.rules_path).ruleset_version,
        "landmark_model": settings.landmark_model_name,
    }
