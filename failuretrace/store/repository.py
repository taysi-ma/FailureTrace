"""The repository — the only write path.

Responsibilities:
- write-once trial persistence to both SQLite and (optionally) raw JSON;
- append-only inserts for hypotheses, promotions, and links (no UPDATE/DELETE);
- the hard-constraint write-time gate that verifies the cross-context justification
  the model alone cannot: (a) deterministic AND repeated failure, (b) an objective
  resource limit exceeded, or (c) effective causal support level >= C2;
- computing a hypothesis's *effective* causal support level from promotions.
"""

from __future__ import annotations

import logging

from ..core.enums import CausalSupportLevel
from ..core.models import (
    DETERMINISTIC_CATEGORIES,
    CounterfactualPlan,
    FailureHypothesis,
    LinkRecord,
    PromotionRecord,
    TrialRecord,
)
from ..core.settings import Settings
from .errors import (
    DuplicateRecordError,
    HardConstraintViolation,
    PromotionViolation,
    ReferentialIntegrityError,
)
from .json_store import JsonStore
from .sqlite_store import SqliteStore

logger = logging.getLogger(__name__)


class Repository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sqlite = SqliteStore(settings)
        self.json = JsonStore(settings)

    # --- trials -----------------------------------------------------------------
    def save_trial(self, rec: TrialRecord) -> TrialRecord:
        """Persist a trial write-once to SQLite and (if enabled) raw JSON."""
        if self.sqlite.get_trial(rec.trial_id) is not None or self.json.exists(rec.trial_id):
            raise DuplicateRecordError(f"trial {rec.trial_id} already exists (write-once)")
        if self.settings.store_raw_json:
            self.json.write_trial(rec)
        self.sqlite.insert_trial(rec)
        logger.info("saved trial %s (status=%s)", rec.trial_id, rec.status)
        return rec

    def get_trial(self, trial_id: str) -> TrialRecord | None:
        return self.sqlite.get_trial(trial_id)

    def list_trials(self) -> list[TrialRecord]:
        return self.sqlite.list_trials()

    def count_trials_for_commit(self, git_commit: str) -> int:
        """How many trials already reference ``git_commit`` (for idempotent ingestion)."""
        return self.sqlite.count_trials_for_commit(git_commit)

    def trial_id_for_commit(self, git_commit: str) -> str | None:
        """The earliest recorded trial id for ``git_commit`` (for parent-lineage linking)."""
        return self.sqlite.trial_id_for_commit(git_commit)

    # --- hypotheses -------------------------------------------------------------
    def save_hypothesis(
        self,
        hyp: FailureHypothesis,
        *,
        telemetry: dict | None = None,
        repeated: bool = False,
        effective_level: CausalSupportLevel | None = None,
        resource_limit_gb: float | None = None,
    ) -> FailureHypothesis:
        """Append a hypothesis, enforcing the hard-constraint justification gate."""
        # Referential integrity: a hypothesis must belong to a persisted trial. The database
        # foreign key (schema v3) also enforces this; the app-level check gives a clear error.
        if not self.sqlite.trial_exists(hyp.trial_id):
            raise ReferentialIntegrityError(
                f"hypothesis {hyp.hypothesis_id} references unknown trial {hyp.trial_id}"
            )
        self._gate_hard_constraint(
            hyp,
            telemetry=telemetry,
            repeated=repeated,
            effective_level=effective_level,
            resource_limit_gb=resource_limit_gb,
        )
        self.sqlite.insert_hypothesis(hyp)
        logger.info(
            "saved hypothesis %s (trial=%s, category=%s, level=%s)",
            hyp.hypothesis_id,
            hyp.trial_id,
            hyp.category,
            hyp.causal_support_level,
        )
        return hyp

    @staticmethod
    def _gate_hard_constraint(
        hyp: FailureHypothesis,
        *,
        telemetry: dict | None,
        repeated: bool,
        effective_level: CausalSupportLevel | None,
        resource_limit_gb: float | None,
    ) -> None:
        if not hyp.should_apply_hard_constraint:
            return
        # (c) effective causal support level >= C2 (via promotions).
        if effective_level is not None and effective_level.at_least(
            CausalSupportLevel.C2_replicated_effect
        ):
            return
        # (b) a configured objective resource limit was exceeded.
        if resource_limit_gb is not None and telemetry:
            peak = telemetry.get("peak_vram_gb")
            if peak is not None and peak >= resource_limit_gb:
                return
        # (a) deterministic AND repeated failure.
        if repeated and hyp.category in DETERMINISTIC_CATEGORIES:
            return
        raise HardConstraintViolation(
            f"hard constraint on hypothesis {hyp.hypothesis_id} is unjustified: requires "
            f"deterministic repeated failure, an exceeded objective resource limit, or C2+"
        )

    def get_hypothesis(self, hypothesis_id: str) -> FailureHypothesis | None:
        return self.sqlite.get_hypothesis(hypothesis_id)

    def list_hypotheses(self) -> list[FailureHypothesis]:
        return self.sqlite.list_hypotheses()

    def list_hypotheses_for_trial(self, trial_id: str) -> list[FailureHypothesis]:
        return self.sqlite.list_hypotheses_for_trial(trial_id)

    # --- promotions & effective level ------------------------------------------
    def save_promotion(self, rec: PromotionRecord) -> PromotionRecord:
        """Append a causal-support promotion, enforcing the write-time evidence gate.

        This makes a promotion non-forgeable regardless of the caller: the referenced
        hypothesis and every supporting trial must exist, ``from_level`` must equal the
        hypothesis's *current* effective level (so the ladder cannot be skipped), and a
        C2 (replication) promotion must carry at least the configured minimum of distinct
        supporting trials. The evaluators in :mod:`failuretrace.planner.replication`
        perform the richer evidence checks; this is the store's defense in depth.
        """
        self._gate_promotion(rec)
        self.sqlite.insert_promotion(rec)
        logger.info(
            "saved promotion %s (%s -> %s) for hypothesis %s",
            rec.promotion_id,
            rec.from_level,
            rec.to_level,
            rec.hypothesis_id,
        )
        return rec

    def _gate_promotion(self, rec: PromotionRecord) -> None:
        from ..classifier.thresholds import load_thresholds  # local: avoid import cycle

        hyp = self.sqlite.get_hypothesis(rec.hypothesis_id)
        if hyp is None:
            raise PromotionViolation(
                f"promotion {rec.promotion_id} references unknown hypothesis {rec.hypothesis_id}"
            )
        # Ladder integrity: you may only promote FROM the current effective level. This
        # structurally forbids skipping a rung (e.g. C1 -> C3 with no C2 in between).
        effective = self.effective_causal_level(rec.hypothesis_id)
        if rec.from_level != effective:
            raise PromotionViolation(
                f"promotion {rec.promotion_id} from_level {rec.from_level.value} != current "
                f"effective level {effective.value if effective else None}"
            )
        # Every supporting / counterfactual trial must be a real persisted trial.
        for tid in list(rec.supporting_trial_ids) + (
            [rec.counterfactual_trial_id] if rec.counterfactual_trial_id else []
        ):
            if not self.sqlite.trial_exists(tid):
                raise PromotionViolation(
                    f"promotion {rec.promotion_id} references unknown trial {tid}"
                )
        # A replication (C2) promotion needs the configured minimum of distinct trials.
        if rec.to_level == CausalSupportLevel.C2_replicated_effect:
            minimum = load_thresholds(self.settings).replication_minimum_trials
            if len(set(rec.supporting_trial_ids)) < minimum:
                raise PromotionViolation(
                    f"promotion {rec.promotion_id} to C2 needs >= {minimum} distinct "
                    f"supporting trials, got {len(set(rec.supporting_trial_ids))}"
                )

    def list_promotions_for_hypothesis(self, hypothesis_id: str) -> list[PromotionRecord]:
        return self.sqlite.list_promotions_for_hypothesis(hypothesis_id)

    def effective_causal_level(self, hypothesis_id: str) -> CausalSupportLevel | None:
        """Original level overridden by the highest valid promotion; None if unknown."""
        hyp = self.sqlite.get_hypothesis(hypothesis_id)
        if hyp is None:
            return None
        level = hyp.causal_support_level
        for promo in self.sqlite.list_promotions_for_hypothesis(hypothesis_id):
            if promo.to_level.rank > level.rank:
                level = promo.to_level
        return level

    # --- links ------------------------------------------------------------------
    def save_link(self, rec: LinkRecord) -> LinkRecord:
        self.sqlite.insert_link(rec)
        return rec

    def list_links_for_hypothesis(self, hypothesis_id: str) -> list[LinkRecord]:
        return self.sqlite.list_links_for_hypothesis(hypothesis_id)

    # --- counterfactual plans ---------------------------------------------------
    def save_plan(self, rec: CounterfactualPlan) -> CounterfactualPlan:
        self.sqlite.insert_plan(rec)
        logger.info("saved counterfactual plan %s for hypothesis %s", rec.plan_id, rec.hypothesis_id)
        return rec

    def get_plan(self, plan_id: str) -> CounterfactualPlan | None:
        return self.sqlite.get_plan(plan_id)

    def list_plans_for_hypothesis(self, hypothesis_id: str) -> list[CounterfactualPlan]:
        return self.sqlite.list_plans_for_hypothesis(hypothesis_id)
