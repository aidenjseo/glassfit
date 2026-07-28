"""End-to-end training-loop smoke test: DB rows -> features -> CV -> artifact -> predict.

Requires pandas + scikit-learn (`uv sync --extra train`); skips cleanly otherwise.
"""

import random
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from schema_builders import sample_frame, sample_measurements, sample_recommendation  # noqa: E402

from glassfit.schemas import CatalogFrame, MatchRatingIn  # noqa: E402
from glassfit.storage.db import connect  # noqa: E402
from glassfit.storage.repo import Repo  # noqa: E402
from glassfit_training import registry  # noqa: E402
from glassfit_training.train import train_rating_model  # noqa: E402

SHAPES = ["rectangle", "round", "panto", "square", "oval", "cat-eye"]


def _frame(i: int, rng: random.Random) -> CatalogFrame:
    return sample_frame(
        frame_id=f"TR-{i:03d}",
        name=f"Trainer {i}",
        shape=rng.choice(SHAPES),
        material=rng.choice(["acetate", "metal", "titanium"]),
        a_mm=rng.uniform(46, 56),
        b_mm=rng.uniform(34, 42),
        dbl_mm=rng.uniform(16, 21),
        ed_mm=rng.uniform(50, 58),
        temple_mm=rng.choice([140.0, 145.0, 150.0]),
        weight_g=rng.uniform(10, 28),
        nose_pads=rng.choice(["fixed_acetate", "adjustable"]),
    )


@pytest.fixture
def ratings_db(tmp_path: Path) -> Path:
    """~90 rating rows whose label depends on shape (a signal fit_score can't see)."""
    rng = random.Random(3)
    path = tmp_path / "train.db"
    conn = connect(path)
    repo = Repo(conn)
    frames = [_frame(i, rng) for i in range(30)]
    repo.upsert_frames(frames)
    for r in range(3):
        rec = sample_recommendation(f"rec-{r}")
        repo.save_recommendation(
            rec,
            scan_id=None,
            user_id="synthetic-test",
            pd_mm=63.0,
            mm_per_unit=0.25,
            measurements=sample_measurements(),
            request_extras={},
        )
        for frame in frames:
            fit = rng.uniform(0.3, 0.9)
            hidden = 0.25 if frame.shape in ("panto", "round") else 0.0
            rating = max(1, min(5, round(1 + 4 * (0.55 * fit + hidden + rng.gauss(0, 0.05)))))
            repo.save_match_rating(
                MatchRatingIn(
                    recommendation_id=rec.recommendation_id,
                    frame_id=frame.frame_id,
                    rating=rating,
                    fit_score=round(fit, 3),
                    components={"bridge_fit": fit, "shape_affinity": rng.uniform(0.4, 1.0)},
                    user_id="synthetic-test",
                )
            )
    conn.close()
    return path


def test_training_loop_end_to_end(
    ratings_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path / "artifacts")
    metrics = train_rating_model(ratings_db, min_rows=50)
    assert metrics is not None
    assert metrics["cv_mae"] < metrics["baseline_mean_mae"]  # learned SOMETHING

    loaded = registry.load_latest("rating")
    assert loaded is not None
    model, metadata = loaded
    assert metadata["target"] == "rating"
    assert metadata["n_rows"] >= 80
    assert "cv_mae" in metadata["metrics"]

    # round-trip prediction on the training features
    from glassfit_training.dataset import load_match_ratings_frame
    from glassfit_training.features import build_rating_features

    frame = load_match_ratings_frame(ratings_db)
    x, _names = build_rating_features(frame)
    preds = model.predict(x.head(5))
    assert len(preds) == 5
    assert all(0.0 <= p <= 6.0 for p in preds)


def test_below_min_rows_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path / "artifacts")
    path = tmp_path / "empty.db"
    connect(path).close()
    assert train_rating_model(path, min_rows=50) is None
    assert registry.load_latest("rating") is None
