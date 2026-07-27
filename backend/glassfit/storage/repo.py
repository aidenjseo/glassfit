"""Repository layer: typed CRUD over the GlassFit SQLite database.

Writes are serialized behind a ``threading.Lock`` (the connection is shared across the FastAPI
threadpool with ``check_same_thread=False``). Reads go straight to SQLite.
"""

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from glassfit.errors import NotFound
from glassfit.schemas import (
    CatalogFrame,
    FaceMeasurements,
    FeedbackIn,
    LandmarkSet,
    Recommendation,
    ScanQuality,
)

_FRAME_UPSERT_SQL = """
INSERT INTO frames (frame_id, name, shape, material, rim, a_mm, b_mm, dbl_mm, ed_mm,
                    temple_mm, weight_g, bridge_style, nose_pads, low_bridge_fit,
                    spring_hinge, default_wrap_deg, tags_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(frame_id) DO UPDATE SET
    name = excluded.name,
    shape = excluded.shape,
    material = excluded.material,
    rim = excluded.rim,
    a_mm = excluded.a_mm,
    b_mm = excluded.b_mm,
    dbl_mm = excluded.dbl_mm,
    ed_mm = excluded.ed_mm,
    temple_mm = excluded.temple_mm,
    weight_g = excluded.weight_g,
    bridge_style = excluded.bridge_style,
    nose_pads = excluded.nose_pads,
    low_bridge_fit = excluded.low_bridge_fit,
    spring_hinge = excluded.spring_hinge,
    default_wrap_deg = excluded.default_wrap_deg,
    tags_json = excluded.tags_json
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_frame(row: sqlite3.Row) -> CatalogFrame:
    return CatalogFrame(
        frame_id=row["frame_id"],
        name=row["name"],
        shape=row["shape"],
        material=row["material"],
        rim=row["rim"],
        a_mm=row["a_mm"],
        b_mm=row["b_mm"],
        dbl_mm=row["dbl_mm"],
        ed_mm=row["ed_mm"],
        temple_mm=row["temple_mm"],
        weight_g=row["weight_g"],
        bridge_style=row["bridge_style"],
        nose_pads=row["nose_pads"],
        low_bridge_fit=bool(row["low_bridge_fit"]),
        spring_hinge=bool(row["spring_hinge"]),
        default_wrap_deg=row["default_wrap_deg"],
        tags=json.loads(row["tags_json"]),
    )


class Repo:
    """All persistence operations for scans, recommendations, frames, and feedback."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._write_lock = threading.Lock()

    # --- scans -------------------------------------------------------------------------------

    def save_scan(
        self,
        scan_id: str,
        landmark_set: LandmarkSet,
        quality: ScanQuality,
        landmark_model: str,
        app_version: str,
    ) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO scans (id, created_at, app_version, landmark_model,"
                " landmarks_json, quality_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    scan_id,
                    _now_iso(),
                    app_version,
                    landmark_model,
                    landmark_set.model_dump_json(),
                    quality.model_dump_json(),
                ),
            )

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "app_version": row["app_version"],
            "landmark_model": row["landmark_model"],
            "landmarks": LandmarkSet.model_validate_json(row["landmarks_json"]),
            "quality": ScanQuality.model_validate_json(row["quality_json"]),
        }

    # --- recommendations ---------------------------------------------------------------------

    def save_recommendation(
        self,
        rec: Recommendation,
        *,
        scan_id: str | None,
        user_id: str,
        pd_mm: float,
        mm_per_unit: float,
        measurements: FaceMeasurements,
        request_extras: dict[str, Any],
        ml_model_version: str | None = None,
    ) -> None:
        """Persist a recommendation. ``request_extras`` must be JSON-serializable
        (rx / lens_intent / pd_monocular as plain dicts)."""
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO recommendations (id, scan_id, created_at, user_id, pd_mm,"
                " mm_per_unit, measurements_json, request_json, output_json, engine_version,"
                " ml_model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rec.recommendation_id,
                    scan_id,
                    _now_iso(),
                    user_id,
                    pd_mm,
                    mm_per_unit,
                    measurements.model_dump_json(),
                    json.dumps(request_extras),
                    rec.model_dump_json(),
                    rec.ruleset_version,
                    ml_model_version,
                ),
            )

    def get_recommendation(self, rec_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM recommendations WHERE id = ?", (rec_id,)).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "scan_id": row["scan_id"],
            "created_at": row["created_at"],
            "user_id": row["user_id"],
            "pd_mm": row["pd_mm"],
            "mm_per_unit": row["mm_per_unit"],
            "measurements": FaceMeasurements.model_validate_json(row["measurements_json"]),
            "request": json.loads(row["request_json"]),
            "recommendation": Recommendation.model_validate_json(row["output_json"]),
            "engine_version": row["engine_version"],
            "ml_model_version": row["ml_model_version"],
        }

    # --- feedback ----------------------------------------------------------------------------

    def save_feedback(self, fb: FeedbackIn) -> str:
        """Persist fit feedback; returns the new feedback id.

        Raises NotFound if the referenced recommendation (or catalog frame, when given)
        does not exist.
        """
        with self._write_lock, self._conn:
            rec = self._conn.execute(
                "SELECT 1 FROM recommendations WHERE id = ?", (fb.recommendation_id,)
            ).fetchone()
            if rec is None:
                raise NotFound(
                    f"recommendation {fb.recommendation_id!r} not found",
                    {"recommendation_id": fb.recommendation_id},
                )
            if fb.frame_id is not None:
                frame = self._conn.execute(
                    "SELECT 1 FROM frames WHERE frame_id = ?", (fb.frame_id,)
                ).fetchone()
                if frame is None:
                    raise NotFound(f"frame {fb.frame_id!r} not found", {"frame_id": fb.frame_id})
            feedback_id = str(uuid.uuid4())
            self._conn.execute(
                "INSERT INTO feedback (id, recommendation_id, frame_id, frame_worn_json,"
                " created_at, user_id, nose_pressure, temple_pressure, slips, cheek_touch,"
                " adjustments_json, comments) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    fb.recommendation_id,
                    fb.frame_id,
                    fb.frame_worn.model_dump_json() if fb.frame_worn is not None else None,
                    _now_iso(),
                    fb.user_id,
                    fb.nose_pressure,
                    fb.temple_pressure,
                    int(fb.slips),
                    int(fb.cheek_touch),
                    fb.adjustments.model_dump_json() if fb.adjustments is not None else None,
                    fb.comments,
                ),
            )
            return feedback_id

    # --- frames ------------------------------------------------------------------------------

    def upsert_frames(self, frames: list[CatalogFrame]) -> None:
        rows = [
            (
                f.frame_id,
                f.name,
                f.shape,
                f.material,
                f.rim,
                f.a_mm,
                f.b_mm,
                f.dbl_mm,
                f.ed_mm,
                f.temple_mm,
                f.weight_g,
                f.bridge_style,
                f.nose_pads,
                int(f.low_bridge_fit),
                int(f.spring_hinge),
                f.default_wrap_deg,
                json.dumps(f.tags),
            )
            for f in frames
        ]
        with self._write_lock, self._conn:
            self._conn.executemany(_FRAME_UPSERT_SQL, rows)

    def list_frames(
        self,
        a_min: float | None = None,
        a_max: float | None = None,
        dbl_min: float | None = None,
        dbl_max: float | None = None,
        temple: float | None = None,
        limit: int = 50,
    ) -> list[CatalogFrame]:
        clauses: list[str] = []
        params: list[Any] = []
        if a_min is not None:
            clauses.append("a_mm >= ?")
            params.append(a_min)
        if a_max is not None:
            clauses.append("a_mm <= ?")
            params.append(a_max)
        if dbl_min is not None:
            clauses.append("dbl_mm >= ?")
            params.append(dbl_min)
        if dbl_max is not None:
            clauses.append("dbl_mm <= ?")
            params.append(dbl_max)
        if temple is not None:
            clauses.append("temple_mm = ?")
            params.append(temple)
        sql = "SELECT * FROM frames"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY frame_id LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_frame(row) for row in rows]

    def get_frame(self, frame_id: str) -> CatalogFrame | None:
        row = self._conn.execute("SELECT * FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
        return None if row is None else _row_to_frame(row)
