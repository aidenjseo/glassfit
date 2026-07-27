"""Application settings. All values overridable via GLASSFIT_* environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GLASSFIT_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    dev: bool = False

    frontend_dir: Path = REPO_ROOT / "frontend"
    db_path: Path = REPO_ROOT / "data" / "runtime" / "glassfit.db"
    models_dir: Path = REPO_ROOT / "data" / "models"
    landmarker_model_path: Path = REPO_ROOT / "data" / "models" / "face_landmarker.task"
    seed_frames_path: Path = REPO_ROOT / "data" / "seed" / "frames.json"
    rules_path: Path = Path(__file__).resolve().parent / "rules" / "defaults.yaml"

    # Debug opt-in ONLY: persist uploaded camera frames to disk (privacy: default off).
    save_frames: bool = False
    save_frames_dir: Path = REPO_ROOT / "data" / "runtime" / "frames"

    # Scan quality gates
    max_yaw_deg: float = 12.0
    max_pitch_deg: float = 12.0
    max_roll_deg: float = 15.0
    min_face_width_frac: float = 0.30  # face bbox width / image width
    min_accepted_frames: int = 3
    max_landmark_dispersion_px: float = 6.0  # above -> quality.ok=False warning (non-blocking)

    landmark_model_name: str = "face_landmarker_v2_478"


@lru_cache
def get_settings() -> Settings:
    return Settings()
