"""Low-level SQLite access: insert-only writes + typed reads.

Concurrency (from the Phase-0 report): autoresearch can run trials in parallel
(per-GPU branches / launcher worktrees). The database is opened in WAL mode with a
busy timeout so concurrent readers are safe; all *writes* are serialized through the
repository. There are deliberately no UPDATE/DELETE methods — records are immutable.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from ..core.models import (
    CounterfactualPlan,
    FailureHypothesis,
    LinkRecord,
    PromotionRecord,
    TrialRecord,
)
from ..core.settings import Settings
from .errors import DuplicateRecordError, ReferentialIntegrityError

logger = logging.getLogger(__name__)

BUSY_TIMEOUT_MS = 5000


def connect(db_path: str | Path, *, enforce_fks: bool = True) -> sqlite3.Connection:
    """Open a connection with WAL + busy timeout and row access by name.

    ``enforce_fks`` toggles ``PRAGMA foreign_keys``. Normal reads/writes enforce foreign
    keys (schema v3) so a dangling reference is rejected at the database level. The schema
    migrator opens with ``enforce_fks=False`` so the v3 table rebuild (drop/rename/copy)
    is not tripped by enforcement; the pragma must be set before any transaction begins.
    """
    conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    conn.execute(f"PRAGMA foreign_keys={'ON' if enforce_fks else 'OFF'};")
    return conn


class SqliteStore:
    """Insert-only CRUD over the FailureTrace tables (hybrid columns + JSON blob)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = Path(settings.paths.data_dir) / "failuretrace.db"

    def _conn(self) -> sqlite3.Connection:
        return connect(self.db_path)

    def _insert(self, sql: str, params: tuple, *, what: str) -> None:
        conn = self._conn()
        try:
            with conn:
                conn.execute(sql, params)
        except sqlite3.IntegrityError as exc:
            # A UNIQUE/PK collision means the append-only record already exists; a FOREIGN
            # KEY failure means it references a parent row that does not — distinct errors.
            if "FOREIGN KEY" in str(exc):
                raise ReferentialIntegrityError(
                    f"{what} references a non-existent parent row"
                ) from exc
            raise DuplicateRecordError(f"{what} already exists (append-only)") from exc
        finally:
            conn.close()

    def _fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = self._conn()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def _fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        conn = self._conn()
        try:
            return conn.execute(sql, params).fetchone()
        finally:
            conn.close()

    # --- trials -----------------------------------------------------------------
    def insert_trial(self, rec: TrialRecord) -> None:
        self._insert(
            "INSERT INTO trials (trial_id, parent_trial_id, status, git_commit, "
            "config_hash, seed, metric_name, timestamp, data) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rec.trial_id,
                rec.parent_trial_id,
                str(rec.status),
                rec.git_commit,
                rec.config_hash,
                rec.seed,
                rec.metric_name,
                rec.timestamp.isoformat(),
                rec.model_dump_json(),
            ),
            what=f"trial {rec.trial_id}",
        )

    def get_trial(self, trial_id: str) -> TrialRecord | None:
        row = self._fetchone("SELECT data FROM trials WHERE trial_id=?", (trial_id,))
        return TrialRecord.model_validate_json(row["data"]) if row else None

    def list_trials(self) -> list[TrialRecord]:
        rows = self._fetchall("SELECT data FROM trials ORDER BY timestamp, trial_id")
        return [TrialRecord.model_validate_json(r["data"]) for r in rows]

    def trial_exists(self, trial_id: str) -> bool:
        return self._fetchone("SELECT 1 FROM trials WHERE trial_id=?", (trial_id,)) is not None

    def count_trials_for_commit(self, git_commit: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM trials WHERE git_commit=?", (git_commit,)
        )
        return int(row["n"]) if row else 0

    # --- hypotheses -------------------------------------------------------------
    def insert_hypothesis(self, rec: FailureHypothesis) -> None:
        self._insert(
            "INSERT INTO hypotheses (hypothesis_id, trial_id, source, category, "
            "causal_support_level, should_apply_soft_penalty, should_apply_hard_constraint, "
            "settings_hash, data) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rec.hypothesis_id,
                rec.trial_id,
                str(rec.source),
                str(rec.category),
                str(rec.causal_support_level),
                int(rec.should_apply_soft_penalty),
                int(rec.should_apply_hard_constraint),
                rec.settings_hash,
                rec.model_dump_json(),
            ),
            what=f"hypothesis {rec.hypothesis_id}",
        )

    def get_hypothesis(self, hypothesis_id: str) -> FailureHypothesis | None:
        row = self._fetchone(
            "SELECT data FROM hypotheses WHERE hypothesis_id=?", (hypothesis_id,)
        )
        return FailureHypothesis.model_validate_json(row["data"]) if row else None

    def list_hypotheses_for_trial(self, trial_id: str) -> list[FailureHypothesis]:
        rows = self._fetchall(
            "SELECT data FROM hypotheses WHERE trial_id=? ORDER BY hypothesis_id",
            (trial_id,),
        )
        return [FailureHypothesis.model_validate_json(r["data"]) for r in rows]

    def list_hypotheses(self) -> list[FailureHypothesis]:
        rows = self._fetchall("SELECT data FROM hypotheses ORDER BY hypothesis_id")
        return [FailureHypothesis.model_validate_json(r["data"]) for r in rows]

    # --- promotions -------------------------------------------------------------
    def insert_promotion(self, rec: PromotionRecord) -> None:
        self._insert(
            "INSERT INTO promotions (promotion_id, hypothesis_id, from_level, to_level, "
            "replication_group_id, counterfactual_trial_id, timestamp, data) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                rec.promotion_id,
                rec.hypothesis_id,
                str(rec.from_level),
                str(rec.to_level),
                rec.replication_group_id,
                rec.counterfactual_trial_id,
                rec.timestamp.isoformat(),
                rec.model_dump_json(),
            ),
            what=f"promotion {rec.promotion_id}",
        )

    def list_promotions_for_hypothesis(self, hypothesis_id: str) -> list[PromotionRecord]:
        rows = self._fetchall(
            "SELECT data FROM promotions WHERE hypothesis_id=? ORDER BY timestamp, promotion_id",
            (hypothesis_id,),
        )
        return [PromotionRecord.model_validate_json(r["data"]) for r in rows]

    # --- links ------------------------------------------------------------------
    def insert_link(self, rec: LinkRecord) -> None:
        self._insert(
            "INSERT INTO links (link_id, link_type, hypothesis_id, trial_id, "
            "source_trial_id, counterfactual_trial_id, replication_group_id, timestamp, data) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rec.link_id,
                str(rec.link_type),
                rec.hypothesis_id,
                rec.trial_id,
                rec.source_trial_id,
                rec.counterfactual_trial_id,
                rec.replication_group_id,
                rec.timestamp.isoformat(),
                rec.model_dump_json(),
            ),
            what=f"link {rec.link_id}",
        )

    def list_links_for_hypothesis(self, hypothesis_id: str) -> list[LinkRecord]:
        rows = self._fetchall(
            "SELECT data FROM links WHERE hypothesis_id=? ORDER BY timestamp, link_id",
            (hypothesis_id,),
        )
        return [LinkRecord.model_validate_json(r["data"]) for r in rows]

    # --- counterfactual plans ---------------------------------------------------
    def insert_plan(self, rec: CounterfactualPlan) -> None:
        self._insert(
            "INSERT INTO plans (plan_id, hypothesis_id, primary_intervention_variable, "
            "coupled_variable, settings_hash, data) VALUES (?,?,?,?,?,?)",
            (
                rec.plan_id,
                rec.hypothesis_id,
                rec.primary_intervention_variable,
                rec.optional_coupled_stabilization_variable,
                rec.settings_hash,
                rec.model_dump_json(),
            ),
            what=f"plan {rec.plan_id}",
        )

    def get_plan(self, plan_id: str) -> CounterfactualPlan | None:
        row = self._fetchone("SELECT data FROM plans WHERE plan_id=?", (plan_id,))
        return CounterfactualPlan.model_validate_json(row["data"]) if row else None

    def list_plans_for_hypothesis(self, hypothesis_id: str) -> list[CounterfactualPlan]:
        rows = self._fetchall(
            "SELECT data FROM plans WHERE hypothesis_id=? ORDER BY plan_id",
            (hypothesis_id,),
        )
        return [CounterfactualPlan.model_validate_json(r["data"]) for r in rows]
