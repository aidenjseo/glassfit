"""Train GlassFit's learning-loop models from the accumulated SQLite data.

    uv run python -m glassfit_training.train [--db PATH] [--min-rows N]

Trains, cross-validates against no-learning baselines, and saves artifacts:
- ``rating``: HistGradientBoostingRegressor predicting the 1-5 match rating from
  the matcher's component snapshot + frame attributes + face measurements.
- ``nose_pressure``/``temple_pressure``/``slips``: comfort models from worn-fit
  feedback (trained only when enough labeled rows exist).

HistGradientBoosting is the deliberate choice for this data regime: small
tabular datasets, mixed numeric/categorical features, NaN-tolerant, no tuning.
"""

import argparse
import sys
from pathlib import Path

from .dataset import load_match_ratings_frame, load_training_frame
from .evaluate import crossval_classification, crossval_regression, rating_heuristic_mae
from .features import build_comfort_features, build_rating_features
from .registry import save_model
from .targets import COMFORT_BINARY, comfort_targets, rating_target

MIN_ROWS_DEFAULT = 60


def _provenance(frame) -> dict:
    return {
        "n_rows": int(len(frame)),
        "label_user_ids": sorted(frame["user_id"].astype(str).unique().tolist()),
        "engine_versions": sorted(frame["engine_version"].astype(str).unique().tolist()),
    }


def train_rating_model(db_path: Path, min_rows: int) -> dict | None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    frame = load_match_ratings_frame(db_path)
    if len(frame) < min_rows:
        print(f"rating: only {len(frame)} rows (< {min_rows}) — skipping")
        return None
    x, feature_names = build_rating_features(frame)
    y = rating_target(frame)

    def factory():
        return HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=7)

    metrics = crossval_regression(factory, x, y)
    heuristic = rating_heuristic_mae(frame, y)
    if heuristic is not None:
        metrics["baseline_fit_score_heuristic_mae"] = heuristic
    model = factory().fit(x, y)
    metadata = {
        "target": "rating",
        "features": feature_names,
        "metrics": metrics,
        **_provenance(frame),
    }
    out = save_model("rating", model, metadata)
    print(
        f"rating: {len(frame)} rows -> CV MAE {metrics['cv_mae']:.3f} "
        f"(mean-baseline {metrics['baseline_mean_mae']:.3f}"
        + (f", fit-score heuristic {heuristic:.3f}" if heuristic is not None else "")
        + f") -> {out}"
    )
    return metrics


def train_comfort_models(db_path: Path, min_rows: int) -> dict[str, dict]:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    frame = load_training_frame(db_path)
    labeled = frame[frame["has_feedback"].astype(bool)]
    results: dict[str, dict] = {}
    if len(labeled) < min_rows:
        print(f"comfort: only {len(labeled)} labeled rows (< {min_rows}) — skipping")
        return results
    x_all, feature_names = build_comfort_features(labeled)
    for name, y in comfort_targets(labeled).items():
        x = x_all.loc[y.index]
        is_binary = name in COMFORT_BINARY
        estimator = HistGradientBoostingClassifier if is_binary else HistGradientBoostingRegressor
        crossval = crossval_classification if is_binary else crossval_regression
        metrics = crossval(lambda: estimator(random_state=7), x, y)  # noqa: B023
        model = estimator(random_state=7).fit(x, y)
        headline = (
            f"CV acc {metrics['cv_accuracy']:.3f} "
            f"(majority {metrics['baseline_majority_accuracy']:.3f})"
            if is_binary
            else (
                f"CV MAE {metrics['cv_mae']:.3f} (mean-baseline {metrics['baseline_mean_mae']:.3f})"
            )
        )
        out = save_model(
            name,
            model,
            {"target": name, "features": feature_names, "metrics": metrics, **_provenance(labeled)},
        )
        print(f"{name}: {len(y)} rows -> {headline} -> {out}")
        results[name] = metrics
    return results


def main() -> None:
    from glassfit.config import get_settings

    parser = argparse.ArgumentParser(prog="glassfit_training.train")
    parser.add_argument("--db", type=Path, default=get_settings().db_path)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS_DEFAULT)
    args = parser.parse_args()

    rating_metrics = train_rating_model(args.db, args.min_rows)
    comfort_metrics = train_comfort_models(args.db, args.min_rows)
    if rating_metrics is None and not comfort_metrics:
        sys.exit("nothing trained — accumulate ratings/feedback first")


if __name__ == "__main__":
    main()
