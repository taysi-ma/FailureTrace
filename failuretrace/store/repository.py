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
    FailureHypothesis,
    LinkRecord,
    PromotionRecord,
    TrialRecord,
)
from ..core.settings import Settings
from .errors import DuplicateRecordError, HardConstraintViolation
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

    # --- promotions & effective level ------------------------------------------
    def save_promotion(self, rec: PromotionRecord) -> PromotionRecord:
        self.sqlite.insert_promotion(rec)
        logger.info(
            "saved promotion %s (%s -> %s) for hypothesis %s",
            rec.promotion_id,
            rec.from_level,
            rec.to_level,
            rec.hypothesis_id,
        )
        return rec

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
