"""Repository layer: typed CRUD over the GlassFit SQLite database.

ALL database access — reads included — is serialized behind one ``threading.Lock``:
the connection is shared across FastAPI's threadpool (``check_same_thread=False``),
and CPython's per-connection statement cache corrupts under concurrent use (probes
showed spurious None rows, NULL columns, and InterfaceError at ~4-12% of overlapped
calls). One lock drove that to zero; the cost is irrelevant for a local app.
"""

import contextlib
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
    MatchRatingIn,
    Recommendation,
    ScanQuality,
)

# Frame column list generated from the schema itself, so adding a CatalogFrame field
# touches exactly one place (the model) plus schema.sql.
_FRAME_COLS = (*(f for f in CatalogFrame.model_fields if f != "tags"), "tags_json")
_FRAME_UPSERT_SQL = (
    f"INSERT INTO frames ({', '.join(_FRAME_COLS)}) "
    f"VALUES ({', '.join(':' + c for c in _FRAME_COLS)}) "
    "ON CONFLICT(frame_id) DO UPDATE SET "
    + ", ".join(f"{c} = excluded.{c}" for c in _FRAME_COLS if c != "frame_id")
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_frame(row: sqlite3.Row) -> CatalogFrame:
    data = dict(row)
    data["tags"] = json.loads(data.pop("tags_json"))
    return CatalogFrame(**data)  # pydantic coerces sqlite's 0/1 back to bool


def _frame_to_row(frame: CatalogFrame) -> dict[str, Any]:
    data = frame.model_dump(exclude={"tags"})
    data["tags_json"] = json.dumps(frame.tags)
    return data


class Repo:
    """All persistence operations for scans, recommendations, frames, and feedback."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.Lock()

    def _checkpoint(self) -> None:
        """Fold the WAL back into the main .db and truncate it (best-effort).

        Privacy: landmark data must never linger in a `-wal` sidecar after a crash —
        the README promises that deleting `data/runtime/` erases everything. Called
        after every write; acquires the lock itself, so call it only OUTSIDE locked
        regions. Failures are swallowed: the next write retries the truncate.
        """
        with self._lock, contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()

    # --- scans -------------------------------------------------------------------------------

    def save_scan(
        self,
        scan_id: str,
        landmark_set: LandmarkSet,
        quality: ScanQuality,
        landmark_model: str,
        app_version: str,
    ) -> None:
        with self._lock, self._conn:
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
        self._checkpoint()

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        with self._lock:
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
        with self._lock, self._conn:
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
        self._checkpoint()

    def get_recommendation(self, rec_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM recommendations WHERE id = ?", (rec_id,)
            ).fetchone()
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
        with self._lock, self._conn:
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
        self._checkpoint()
        return feedback_id

    # --- match ratings -----------------------------------------------------------------------

    def save_match_rating(self, rating: MatchRatingIn) -> str:
        """Persist a per-frame match rating; returns the new rating id.

        Raises NotFound when the referenced recommendation or catalog frame is unknown.
        """
        with self._lock, self._conn:
            if (
                self._conn.execute(
                    "SELECT 1 FROM recommendations WHERE id = ?", (rating.recommendation_id,)
                ).fetchone()
                is None
            ):
                raise NotFound(
                    f"recommendation {rating.recommendation_id!r} not found",
                    {"recommendation_id": rating.recommendation_id},
                )
            if (
                self._conn.execute(
                    "SELECT 1 FROM frames WHERE frame_id = ?", (rating.frame_id,)
                ).fetchone()
                is None
            ):
                raise NotFound(
                    f"frame {rating.frame_id!r} not found", {"frame_id": rating.frame_id}
                )
            # Upsert: re-rating the same (recommendation, frame, user) replaces the
            # previous label — latest opinion wins; RETURNING yields the surviving id.
            row = self._conn.execute(
                "INSERT INTO match_ratings (id, recommendation_id, frame_id, created_at,"
                " user_id, rating, fit_score, components_json, comment)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(recommendation_id, frame_id, user_id) DO UPDATE SET"
                " rating = excluded.rating, fit_score = excluded.fit_score,"
                " components_json = excluded.components_json, comment = excluded.comment,"
                " created_at = excluded.created_at"
                " RETURNING id",
                (
                    str(uuid.uuid4()),
                    rating.recommendation_id,
                    rating.frame_id,
                    _now_iso(),
                    rating.user_id,
                    rating.rating,
                    rating.fit_score,
                    json.dumps(rating.components) if rating.components is not None else None,
                    rating.comment,
                ),
            ).fetchone()
        self._checkpoint()
        return row["id"]

    # --- frames ------------------------------------------------------------------------------

    def upsert_frames(self, frames: list[CatalogFrame]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(_FRAME_UPSERT_SQL, [_frame_to_row(f) for f in frames])

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
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_frame(row) for row in rows]

    def get_frame(self, frame_id: str) -> CatalogFrame | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM frames WHERE frame_id = ?", (frame_id,)
            ).fetchone()
        return None if row is None else _row_to_frame(row)
