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

# v3: add referential integrity. SQLite cannot ALTER TABLE ADD CONSTRAINT, so each child
# table is rebuilt (create-with-FK, copy, drop, rename, re-index). Applied with foreign-key
# enforcement OFF (see initialize_database) so the drop/rename ordering is not tripped;
# subsequent normal connections open with enforcement ON. Parents are rebuilt before the
# children that reference them. Legacy rows are copied verbatim (not re-validated).
_DDL_V3 = """
CREATE TABLE hypotheses_v3 (
    hypothesis_id                TEXT PRIMARY KEY,
    trial_id                     TEXT NOT NULL,
    source                       TEXT NOT NULL,
    category                     TEXT NOT NULL,
    causal_support_level         TEXT NOT NULL,
    should_apply_soft_penalty    INTEGER NOT NULL,
    should_apply_hard_constraint INTEGER NOT NULL,
    settings_hash                TEXT NOT NULL,
    data                         TEXT NOT NULL,
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);
INSERT INTO hypotheses_v3 SELECT
    hypothesis_id, trial_id, source, category, causal_support_level,
    should_apply_soft_penalty, should_apply_hard_constraint, settings_hash, data
    FROM hypotheses;
DROP TABLE hypotheses;
ALTER TABLE hypotheses_v3 RENAME TO hypotheses;
CREATE INDEX IF NOT EXISTS idx_hypotheses_trial    ON hypotheses(trial_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_category ON hypotheses(category);

CREATE TABLE promotions_v3 (
    promotion_id            TEXT PRIMARY KEY,
    hypothesis_id           TEXT NOT NULL,
    from_level              TEXT NOT NULL,
    to_level                TEXT NOT NULL,
    replication_group_id    TEXT,
    counterfactual_trial_id TEXT,
    timestamp               TEXT NOT NULL,
    data                    TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id),
    FOREIGN KEY (counterfactual_trial_id) REFERENCES trials(trial_id)
);
INSERT INTO promotions_v3 SELECT
    promotion_id, hypothesis_id, from_level, to_level, replication_group_id,
    counterfactual_trial_id, timestamp, data
    FROM promotions;
DROP TABLE promotions;
ALTER TABLE promotions_v3 RENAME TO promotions;
CREATE INDEX IF NOT EXISTS idx_promotions_hyp ON promotions(hypothesis_id);

CREATE TABLE plans_v3 (
    plan_id                       TEXT PRIMARY KEY,
    hypothesis_id                 TEXT NOT NULL,
    primary_intervention_variable TEXT NOT NULL,
    coupled_variable              TEXT,
    settings_hash                 TEXT NOT NULL,
    data                          TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);
INSERT INTO plans_v3 SELECT
    plan_id, hypothesis_id, primary_intervention_variable, coupled_variable,
    settings_hash, data
    FROM plans;
DROP TABLE plans;
ALTER TABLE plans_v3 RENAME TO plans;
CREATE INDEX IF NOT EXISTS idx_plans_hyp ON plans(hypothesis_id);

CREATE TABLE links_v3 (
    link_id                 TEXT PRIMARY KEY,
    link_type               TEXT NOT NULL,
    hypothesis_id           TEXT,
    trial_id                TEXT,
    source_trial_id         TEXT,
    counterfactual_trial_id TEXT,
    replication_group_id    TEXT,
    timestamp               TEXT NOT NULL,
    data                    TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id),
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);
INSERT INTO links_v3 SELECT
    link_id, link_type, hypothesis_id, trial_id, source_trial_id,
    counterfactual_trial_id, replication_group_id, timestamp, data
    FROM links;
DROP TABLE links;
ALTER TABLE links_v3 RENAME TO links;
CREATE INDEX IF NOT EXISTS idx_links_hyp ON links(hypothesis_id);
"""

# v4: make append-only immutability *physical*. Records are inserted once and never
# updated or deleted; these BEFORE UPDATE/DELETE triggers raise, so a stray raw-SQL
# mutation is rejected by the database, not merely by API discipline. (INSERT is
# unaffected — promotions and links are still plain inserts.)
_IMMUTABLE_TABLES = ("trials", "hypotheses", "promotions", "plans", "links")
_DDL_V4 = "\n".join(
    f"""
CREATE TRIGGER IF NOT EXISTS trg_{t}_no_update BEFORE UPDATE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} are append-only / immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_{t}_no_delete BEFORE DELETE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} are append-only / immutable'); END;
"""
    for t in _IMMUTABLE_TABLES
)

# v5: persist the deterministic classifier output as its own immutable, trial-linked record.
# The classifier's category, confidence, and triggered rules survive verbatim even when the
# optional LLM later overwrites the hypothesis narrative (independent provenance).
_DDL_V5 = """
CREATE TABLE classifications (
    trial_id      TEXT PRIMARY KEY,
    category      TEXT NOT NULL,
    confidence    REAL NOT NULL,
    settings_hash TEXT NOT NULL,
    data          TEXT NOT NULL,
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);
CREATE TRIGGER IF NOT EXISTS trg_classifications_no_update BEFORE UPDATE ON classifications
BEGIN SELECT RAISE(ABORT, 'classifications are append-only / immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_classifications_no_delete BEFORE DELETE ON classifications
BEGIN SELECT RAISE(ABORT, 'classifications are append-only / immutable'); END;
"""

# (version, ddl) applied in ascending order. Append new steps; never edit shipped ones.
SCHEMA_STEPS: list[tuple[int, str]] = [
    (1, _DDL_V1),
    (2, _DDL_V2),
    (3, _DDL_V3),
    (4, _DDL_V4),
    (5, _DDL_V5),
]


def initialize_database(settings: Settings) -> Path:
    """Create/upgrade the database. Idempotent; returns the database path."""
    data_dir = Path(settings.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "failuretrace.db"

    # Enforcement OFF during migration so the v3 table rebuild (drop/rename/copy) is not
    # tripped by foreign keys; the pragma must be set before any transaction begins.
    conn = connect(db_path, enforce_fks=False)
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
