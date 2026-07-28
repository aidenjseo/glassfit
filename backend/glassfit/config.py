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
    # Optional local try-on product art (<frame_id>.png), personal use, never committed
    tryon_dir: Path = REPO_ROOT / "data" / "tryon"
    rules_path: Path = Path(__file__).resolve().parent / "rules" / "defaults.yaml"

    # Debug opt-in ONLY: persist uploaded camera frames to disk (privacy: default off).
    save_frames: bool = False
    save_frames_dir: Path = REPO_ROOT / "data" / "runtime" / "frames"

    # Scan quality gates (frontal burst). Yaw/roll distort the width measurements the
    # rules engine depends on, so they stay strict. Pitch barely affects x-separations
    # and real-world captures (webcam below eye level, portrait chin angles) routinely
    # read +10-20° — calibrated against real portrait photos.
    max_yaw_deg: float = 12.0
    max_pitch_deg: float = 20.0
    max_roll_deg: float = 15.0
    # Calibrated against mediapipe's official portrait test image (0.23): a well-framed
    # headshot at laptop distance spans ~0.15-0.30 of the frame width.
    min_face_width_frac: float = 0.14
    min_accepted_frames: int = 3
    max_landmark_dispersion_px: float = 6.0  # above -> quality.ok=False warning (non-blocking)
    # Side-view (head-turn) bursts: must show a genuine turn, but not a full profile
    min_side_yaw_deg: float = 10.0
    max_side_yaw_deg: float = 60.0

    # Live distance-coaching bands (face width / image width). The hard scan gate is
    # min_face_width_frac; the probe guides users into the comfortable middle.
    probe_ideal_min_frac: float = 0.20
    probe_ideal_max_frac: float = 0.45

    landmark_model_name: str = "face_landmarker_v2_478"


@lru_cache
def get_settings() -> Settings:
    return Settings()
