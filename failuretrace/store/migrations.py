"""Idempotent schema initialization — a lightweight, framework-free migrator.

``initialize_database`` creates a ``schema_version`` table and applies numbered DDL
steps that have not yet been applied. Calling it repeatedly is a no-op after the first
run (each version is applied at most once).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from ..core.settings import Settings
from .sqlite_store import connect

logger = logging.getLogger(__name__)

_DDL_V1 = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id        TEXT PRIMARY KEY,
    parent_trial_id TEXT,
    status          TEXT NOT NULL,
    git_commit      TEXT,
    config_hash     TEXT,
    seed            INTEGER,
    metric_name     TEXT,
    timestamp       TEXT NOT NULL,
    data            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id                TEXT PRIMARY KEY,
    trial_id                     TEXT NOT NULL,
    source                       TEXT NOT NULL,
    category                     TEXT NOT NULL,
    causal_support_level         TEXT NOT NULL,
    should_apply_soft_penalty    INTEGER NOT NULL,
    should_apply_hard_constraint INTEGER NOT NULL,
    settings_hash                TEXT NOT NULL,
    data                         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotions (
    promotion_id            TEXT PRIMARY KEY,
    hypothesis_id           TEXT NOT NULL,
    from_level              TEXT NOT NULL,
    to_level                TEXT NOT NULL,
    replication_group_id    TEXT,
    counterfactual_trial_id TEXT,
    timestamp               TEXT NOT NULL,
    data                    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS links (
    link_id                 TEXT PRIMARY KEY,
    link_type               TEXT NOT NULL,
    hypothesis_id           TEXT,
    trial_id                TEXT,
    source_trial_id         TEXT,
    counterfactual_trial_id TEXT,
    replication_group_id    TEXT,
    timestamp               TEXT NOT NULL,
    data                    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_trial    ON hypotheses(trial_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_category ON hypotheses(category);
CREATE INDEX IF NOT EXISTS idx_promotions_hyp      ON promotions(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_links_hyp           ON links(hypothesis_id);
"""

_DDL_V2 = """
CREATE TABLE IF NOT EXISTS plans (
    plan_id                       TEXT PRIMARY KEY,
    hypothesis_id                 TEXT NOT NULL,
    primary_intervention_variable TEXT NOT NULL,
    coupled_variable              TEXT,
    settings_hash                 TEXT NOT NULL,
    data                          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plans_hyp ON plans(hypothesis_id);
"""

# (version, ddl) applied in ascending order. Append new steps; never edit shipped ones.
SCHEMA_STEPS: list[tuple[int, str]] = [
    (1, _DDL_V1),
    (2, _DDL_V2),
]


def initialize_database(settings: Settings) -> Path:
    """Create/upgrade the database. Idempotent; returns the database path."""
    data_dir = Path(settings.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "failuretrace.db"

    conn = connect(db_path)
    try:
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            current = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM schema_version"
            ).fetchone()["v"]
            for version, ddl in SCHEMA_STEPS:
                if version > current:
                    conn.executescript(ddl)
                    conn.execute(
                        "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                        (version, datetime.now(timezone.utc).isoformat()),
                    )
                    logger.info("applied schema version %d", version)
    finally:
        conn.close()
    return db_path
