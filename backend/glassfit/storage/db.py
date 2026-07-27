"""SQLite connection factory with PRAGMA user_version migrations.

The connection is created with ``check_same_thread=False`` because FastAPI runs sync endpoints
in a threadpool; callers (``Repo``) serialize writes behind a lock.
"""

import sqlite3
from collections.abc import Callable
from pathlib import Path

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

Migration = Callable[[sqlite3.Connection], None]


def _apply_base_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))


# Ordered migrations: (target user_version, migration to reach it from the previous version).
# To evolve the schema, append e.g. ``(2, _migrate_v1_to_v2)`` — never edit shipped entries.
MIGRATIONS: tuple[tuple[int, Migration], ...] = ((1, _apply_base_schema),)


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
