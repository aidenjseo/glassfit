"""Seed-catalog loading: validate data/seed/frames.json and upsert it at startup."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from glassfit.schemas import CatalogFrame
from glassfit.storage.repo import Repo

_FRAME_LIST = TypeAdapter(list[CatalogFrame])


def load_seed(path: Path) -> list[CatalogFrame]:
    """Load and pydantic-validate the seed frame catalog (a JSON array of CatalogFrame)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _FRAME_LIST.validate_python(raw)


def ensure_seeded(repo: Repo, path: Path) -> None:
    """Idempotently upsert the seed catalog into the frames table (call at app startup)."""
    repo.upsert_frames(load_seed(path))
