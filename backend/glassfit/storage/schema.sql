-- GlassFit SQLite schema (authoritative DDL, schema version 1).
-- Timestamps are ISO-8601 UTC strings. Sides are subject-anatomical (right = OD).

CREATE TABLE IF NOT EXISTS scans (
    id             TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL, -- ISO-8601 UTC
    app_version    TEXT NOT NULL,
    landmark_model TEXT NOT NULL,
    landmarks_json TEXT NOT NULL, -- serialized LandmarkSet
    quality_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
    id                TEXT PRIMARY KEY,
    scan_id           TEXT REFERENCES scans(id),
    created_at        TEXT NOT NULL,
    user_id           TEXT NOT NULL DEFAULT 'local',
    pd_mm             REAL NOT NULL,
    mm_per_unit       REAL NOT NULL,
    measurements_json TEXT NOT NULL, -- FaceMeasurements dump
    request_json      TEXT NOT NULL, -- rx, lens_intent, pd_monocular
    output_json       TEXT NOT NULL, -- Recommendation dump incl rule_trace
    engine_version    TEXT NOT NULL,
    ml_model_version  TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    frame_id         TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    shape            TEXT NOT NULL,
    material         TEXT NOT NULL,
    rim              TEXT NOT NULL,
    a_mm             REAL NOT NULL,
    b_mm             REAL NOT NULL,
    dbl_mm           REAL NOT NULL,
    ed_mm            REAL NOT NULL,
    temple_mm        REAL NOT NULL,
    weight_g         REAL NOT NULL,
    bridge_style     TEXT NOT NULL,
    nose_pads        TEXT NOT NULL,
    low_bridge_fit   INTEGER NOT NULL DEFAULT 0,
    spring_hinge     INTEGER NOT NULL DEFAULT 0,
    default_wrap_deg REAL,
    tags_json        TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS feedback (
    id                TEXT PRIMARY KEY,
    recommendation_id TEXT NOT NULL REFERENCES recommendations(id),
    frame_id          TEXT REFERENCES frames(frame_id),
    frame_worn_json   TEXT,
    created_at        TEXT NOT NULL,
    user_id           TEXT NOT NULL DEFAULT 'local',
    nose_pressure     INTEGER NOT NULL CHECK (nose_pressure BETWEEN 1 AND 5),
    temple_pressure   INTEGER NOT NULL CHECK (temple_pressure BETWEEN 1 AND 5),
    slips             INTEGER NOT NULL CHECK (slips IN (0, 1)),
    cheek_touch       INTEGER NOT NULL CHECK (cheek_touch IN (0, 1)),
    adjustments_json  TEXT,
    comments          TEXT
);

CREATE INDEX IF NOT EXISTS idx_recommendations_scan_id ON recommendations(scan_id);
CREATE INDEX IF NOT EXISTS idx_feedback_recommendation_id ON feedback(recommendation_id);

PRAGMA user_version = 1;
