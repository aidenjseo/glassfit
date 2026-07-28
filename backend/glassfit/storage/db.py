"""SQLite connection factory with PRAGMA user_version migrations.

The connection is created with ``check_same_thread=False`` because FastAPI runs sync endpoints
in a threadpool; callers (``Repo``) serialize writes behind a lock.
"""

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

Migration = Callable[[sqlite3.Connection], None]


def _apply_base_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))


_MATCH_RATINGS_DDL = """
CREATE TABLE IF NOT EXISTS match_ratings (
  id                TEXT PRIMARY KEY,
  recommendation_id TEXT NOT NULL REFERENCES recommendations(id),
  frame_id          TEXT NOT NULL REFERENCES frames(frame_id),
  created_at        TEXT NOT NULL,
  user_id           TEXT NOT NULL DEFAULT 'local',
  rating            INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
  fit_score         REAL,
  components_json   TEXT,
  comment           TEXT
);
CREATE INDEX IF NOT EXISTS idx_match_ratings_rec ON match_ratings(recommendation_id);
"""


def _add_match_ratings(conn: sqlite3.Connection) -> None:
    """v2: per-frame match ratings — training labels for the matching algorithm."""
    conn.executescript(_MATCH_RATINGS_DDL)


def _unique_match_ratings(conn: sqlite3.Connection) -> None:
    """v3: one rating per (recommendation, frame, user) — latest wins.

    Deduplicates any pre-existing rows, then enforces uniqueness so re-rating
    upserts instead of stacking contradictory training labels.
    """
    conn.executescript(
        """
        DELETE FROM match_ratings WHERE rowid NOT IN (
            SELECT MAX(rowid) FROM match_ratings
            GROUP BY recommendation_id, frame_id, user_id
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_match_ratings_pair
            ON match_ratings(recommendation_id, frame_id, user_id);
        """
    )


def _backfill_face_length_jaw(conn: sqlite3.Connection) -> None:
    """v4: backfill ``face_length_mm``/``jaw_width_mm`` into pre-existing rows.

    The fields were added to FaceMeasurements after early rows were written; without
    them, re-parsing legacy ``measurements_json`` fails. Backfilled values are
    ESTIMATES from the empirical proportion means of 77 real portrait scans
    (length ~1.20 x zygoma, jaw ~0.91 x zygoma; see catalog.match EMPIRICAL_* for
    the LIVE constants — these literals are frozen with the shipped migration)
    and are flagged in warnings.
    """
    rows = conn.execute("SELECT id, measurements_json FROM recommendations").fetchall()
    for row in rows:
        data = json.loads(row[1])
        if "face_length_mm" in data and "jaw_width_mm" in data:
            continue
        zygoma = float(data.get("zygoma_width_mm", 130.0))
        data.setdefault("face_length_mm", round(1.20 * zygoma, 2))
        data.setdefault("jaw_width_mm", round(0.91 * zygoma, 2))
        quality = data.setdefault("quality", {})
        quality.setdefault("warnings", []).append("face_length_jaw_backfilled_v4")
        conn.execute(
            "UPDATE recommendations SET measurements_json = ? WHERE id = ?",
            (json.dumps(data), row[0]),
        )


# Ordered migrations: (target user_version, migration to reach it from the previous version).
# To evolve the schema, append the next entry — never edit shipped entries.
MIGRATIONS: tuple[tuple[int, Migration], ...] = (
    (1, _apply_base_schema),
    (2, _add_match_ratings),
    (3, _unique_match_ratings),
    (4, _backfill_face_length_jaw),
)


def _migrate(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    for target, migration in MIGRATIONS:
        if version < target:
            migration(conn)
            conn.execute(f"PRAGMA user_version = {target:d}")
            conn.commit()
            version = target


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the GlassFit database and bring it to the current schema."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _migrate(conn)
    return conn
