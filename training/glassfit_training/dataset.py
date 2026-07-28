"""Training-set assembly: SQLite -> joined, flattened pandas DataFrame -> CSV/parquet.

Implemented in Phase 1 so data accumulates with full provenance from day one.
Row grain: one row per (recommendation, feedback) pair; recommendations without
feedback are kept (LEFT JOIN) and flagged ``has_feedback=False``.

Column families:
- ``m_*``     flattened FaceMeasurements (nested objects joined with underscores)
- ``rec_*``   flattened Recommendation outputs (``rule_trace``/``notes`` excluded)
- ``adj_*``   flattened feedback adjustment deltas (the residual-learning targets)
- labels:     nose_pressure, temple_pressure, slips, cheek_touch, comments
- provenance: engine_version, ml_model_version, landmark_model, app_version, user_id, ...

CLI:  uv run python -m glassfit_training.dataset export [--db PATH] [--out PATH] [--fmt csv|parquet]
"""

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_JOIN_SQL = """
SELECT
    s.id              AS scan_id,
    s.created_at      AS scan_created_at,
    s.app_version     AS app_version,
    s.landmark_model  AS landmark_model,
    r.id              AS recommendation_id,
    r.created_at      AS recommended_at,
    r.user_id         AS user_id,
    r.pd_mm           AS pd_mm,
    r.mm_per_unit     AS mm_per_unit,
    r.measurements_json,
    r.output_json,
    r.engine_version  AS engine_version,
    r.ml_model_version AS ml_model_version,
    f.id              AS feedback_id,
    f.created_at      AS feedback_at,
    f.frame_id        AS frame_id,
    f.frame_worn_json,
    f.nose_pressure, f.temple_pressure, f.slips, f.cheek_touch,
    f.adjustments_json, f.comments
FROM recommendations r
LEFT JOIN scans s    ON s.id = r.scan_id
LEFT JOIN feedback f ON f.recommendation_id = r.id
ORDER BY r.created_at, f.created_at
"""

_RATINGS_SQL = """
SELECT
    mr.id              AS rating_id,
    mr.recommendation_id,
    mr.frame_id,
    mr.created_at      AS rated_at,
    mr.user_id         AS user_id,
    mr.rating          AS rating,
    mr.fit_score       AS fit_score,
    mr.components_json,
    mr.comment         AS comment,
    r.measurements_json,
    r.engine_version   AS engine_version,
    r.pd_mm            AS pd_mm,
    f.a_mm             AS frame_a_mm,
    f.b_mm             AS frame_b_mm,
    f.dbl_mm           AS frame_dbl_mm,
    f.ed_mm            AS frame_ed_mm,
    f.temple_mm        AS frame_temple_mm,
    f.weight_g         AS frame_weight_g,
    f.shape            AS frame_shape,
    f.material         AS frame_material,
    f.nose_pads        AS frame_nose_pads,
    f.low_bridge_fit   AS frame_low_bridge_fit,
    f.spring_hinge     AS frame_spring_hinge
FROM match_ratings mr
JOIN recommendations r ON r.id = mr.recommendation_id
JOIN frames f          ON f.frame_id = mr.frame_id
ORDER BY mr.created_at
"""

_SKIP_KEYS = {"rule_trace", "notes"}


def _require_pandas():
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "pandas is required for dataset export — install with `uv sync --extra train`"
        ) from exc
    return pandas


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    """Flatten nested dicts of scalars into underscore-joined columns."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _SKIP_KEYS:
                continue
            _flatten(f"{prefix}_{key}", value, out)
    elif isinstance(obj, bool | int | float | str) or obj is None:
        out[prefix] = obj
    # lists (e.g. warnings) are intentionally dropped


def load_training_frame(db_path: Path):
    """Load the joined training table as a pandas DataFrame."""
    pd = _require_pandas()
    if not Path(db_path).exists():
        raise SystemExit(f"database not found: {db_path} — run the app and scan first")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(_JOIN_SQL).fetchall()
    finally:
        conn.close()

    records: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {
            "recommendation_id": row["recommendation_id"],
            "scan_id": row["scan_id"],
            "user_id": row["user_id"],
            "pd_mm": row["pd_mm"],
            "mm_per_unit": row["mm_per_unit"],
            "engine_version": row["engine_version"],
            "ml_model_version": row["ml_model_version"],
            "landmark_model": row["landmark_model"],
            "app_version": row["app_version"],
            "recommended_at": row["recommended_at"],
            "has_feedback": row["feedback_id"] is not None,
            "feedback_id": row["feedback_id"],
            "frame_id": row["frame_id"],
            "nose_pressure": row["nose_pressure"],
            "temple_pressure": row["temple_pressure"],
            "slips": row["slips"],
            "cheek_touch": row["cheek_touch"],
            "comments": row["comments"],
        }
        _flatten("m", json.loads(row["measurements_json"]), rec)
        _flatten("rec", json.loads(row["output_json"]), rec)
        if row["frame_worn_json"]:
            _flatten("worn", json.loads(row["frame_worn_json"]), rec)
        if row["adjustments_json"]:
            _flatten("adj", json.loads(row["adjustments_json"]), rec)
        records.append(rec)
    return pd.DataFrame(records)


def load_match_ratings_frame(db_path: Path):
    """Frame-match ratings joined with measurements + frame attributes.

    The learning table for the MATCHING algorithm: features = flattened
    measurements (``m_*``), catalog-frame attributes (``frame_*``), and the
    matcher's component snapshot at rating time (``comp_*``); label = ``rating``.
    """
    pd = _require_pandas()
    if not Path(db_path).exists():
        raise SystemExit(f"database not found: {db_path} — run the app and rate matches first")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(_RATINGS_SQL).fetchall()
    finally:
        conn.close()

    records: list[dict[str, Any]] = []
    for row in rows:
        rec = {
            key: row[key]
            for key in row.keys()  # noqa: SIM118 - sqlite3.Row has no __iter__ over keys
            if key not in ("measurements_json", "components_json")
        }
        _flatten("m", json.loads(row["measurements_json"]), rec)
        if row["components_json"]:
            _flatten("comp", json.loads(row["components_json"]), rec)
        records.append(rec)
    return pd.DataFrame(records)


_LOADERS = {"feedback": load_training_frame, "ratings": load_match_ratings_frame}


def export(db_path: Path, out_path: Path, fmt: str = "csv", table: str = "feedback") -> Path:
    """Write a training table to ``out_path`` (csv or parquet); returns the path."""
    frame = _LOADERS[table](db_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            frame.to_parquet(out_path)
            return out_path
        except ImportError:
            out_path = out_path.with_suffix(".csv")
            print("pyarrow not installed — falling back to CSV")
    frame.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    from glassfit.config import get_settings

    parser = argparse.ArgumentParser(prog="glassfit_training.dataset")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="export a joined training table")
    exp.add_argument("--db", type=Path, default=get_settings().db_path)
    exp.add_argument("--out", type=Path, default=None)
    exp.add_argument("--fmt", choices=["csv", "parquet"], default="csv")
    exp.add_argument(
        "--table",
        choices=sorted(_LOADERS),
        default="feedback",
        help="feedback = worn-fit comfort labels; ratings = frame-match ratings",
    )
    args = parser.parse_args()

    out = args.out
    if out is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        from glassfit.config import REPO_ROOT

        out = REPO_ROOT / "data" / "exports" / f"training_{args.table}_{stamp}.{args.fmt}"
    written = export(args.db, out, args.fmt, table=args.table)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
