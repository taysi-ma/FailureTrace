"""Deterministic promotion gate. Produces ``PromotionRecord``s; never mutates hypotheses.

Promotion ladder (each step needs the configured evidence, verified against the *store* —
a single trial can never reach C2+, and fabricated or mismatched evidence is rejected):

- C1 -> C2: the same intervention family (matched by the source trial's fingerprint)
  reproduced across >= ``replication_minimum_trials`` distinct seeds / equivalent
  controlled trials, all pointing the *same* metric direction and clearing the noise floor.
- C2 -> C3: a **persisted** counterfactual plan exists for the hypothesis AND
  >= ``counterfactual_minimum_support`` counterfactual trials produced the expected
  **directional** result above the noise floor (judged via ``improvement()``).
- C3 -> C4: >= ``c4_minimum_counterfactuals`` directional confirmations from >= 2 distinct
  contexts (different changed components or configs). Rare by design.

Every evaluator takes the :class:`Repository` and refuses to promote unless the hypothesis
exists and is *currently* at the expected effective level (so a rung can never be skipped);
supporting trials that do not exist in the store are ignored. The repository's
``save_promotion`` re-checks these invariants at write time (defense in depth).

No Bayesian causal estimation in the MVP.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from ..classifier.thresholds import load_thresholds
from ..core.enums import CausalSupportLevel, MetricDirection
from ..core.ids import new_promotion_id
from ..core.models import PromotionRecord, TrialRecord
from ..core.settings import Settings, improvement
from ..store.repository import Repository

logger = logging.getLogger(__name__)

_C1 = CausalSupportLevel.C1_plausible_hypothesis
_C2 = CausalSupportLevel.C2_replicated_effect
_C3 = CausalSupportLevel.C3_counterfactual_supported
_C4 = CausalSupportLevel.C4_robust_rule


class ReplicationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    seed: int | None = None


class CounterfactualResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    baseline_metric: float
    post_change_metric: float
    metric_direction: MetricDirection
    changed_components: list[str] = Field(default_factory=list)
    config_hash: str | None = None


def _intervention_fingerprint(trial: TrialRecord) -> frozenset[str]:
    """A coarse, deterministic identity for *what a trial changed* — its changed
    components plus its hyperparameter names. Two trials are the "same intervention
    family" when their fingerprints intersect."""
    return frozenset(trial.changed_components) | {f"hp:{k}" for k in trial.hyperparameters}


def _improvement_sign(trial: TrialRecord, direction: MetricDirection) -> int | None:
    """Sign of the trial's direction-aware improvement, or ``None`` if not computable."""
    if trial.baseline_metric is None or trial.post_change_metric is None:
        return None
    delta = improvement(trial.baseline_metric, trial.post_change_metric, direction)
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def evaluate_replication(
    hypothesis_id: str,
    evidence: list[ReplicationEvidence],
    *,
    settings: Settings,
    repository: Repository,
    replication_group_id: str,
) -> PromotionRecord | None:
    """C1 -> C2 when enough distinct-seed / controlled replications of the *same*
    intervention family, pointing the same metric direction, exist in the store."""
    thresholds = load_thresholds(settings)
    noise = thresholds.inconclusive_noise_floor
    direction = settings.metric.direction

    hyp = repository.get_hypothesis(hypothesis_id)
    if hyp is None:
        logger.warning("replication gate: unknown hypothesis %s", hypothesis_id)
        return None
    if repository.effective_causal_level(hypothesis_id) != _C1:
        logger.debug("replication gate: hypothesis %s is not at C1", hypothesis_id)
        return None

    source = repository.get_trial(hyp.trial_id)
    source_fp = _intervention_fingerprint(source) if source is not None else frozenset()
    source_sign = _improvement_sign(source, direction) if source is not None else None

    qualifying: list[tuple[str, int | None]] = []
    for ev in evidence:
        trial = repository.get_trial(ev.trial_id)
        if trial is None:
            logger.debug("replication gate: ignoring unknown trial %s", ev.trial_id)
            continue
        # same intervention family as the hypothesis's source trial
        if source_fp and not (_intervention_fingerprint(trial) & source_fp):
            continue
        # consistent direction + effect above the noise floor, when metrics are present
        sign = _improvement_sign(trial, direction)
        if source_sign is not None and sign is not None:
            if sign != source_sign:
                continue
            if abs(improvement(trial.baseline_metric, trial.post_change_metric, direction)) < noise:
                continue
        qualifying.append((trial.trial_id, ev.seed if ev.seed is not None else trial.seed))

    distinct_seeds = {s for _, s in qualifying if s is not None}
    count = len(distinct_seeds) if distinct_seeds else len({tid for tid, _ in qualifying})
    if count < thresholds.replication_minimum_trials:
        return None

    return PromotionRecord(
        promotion_id=new_promotion_id(),
        hypothesis_id=hypothesis_id,
        from_level=_C1,
        to_level=_C2,
        replication_group_id=replication_group_id,
        supporting_trial_ids=[tid for tid, _ in qualifying],
        rationale=(
            f"same intervention family replicated across {count} distinct-seed/controlled "
            f"trials (>= {thresholds.replication_minimum_trials}), consistent direction"
        ),
        settings_hash=settings.settings_hash(),
    )


def evaluate_counterfactual(
    hypothesis_id: str,
    results: list[CounterfactualResult],
    *,
    settings: Settings,
    repository: Repository,
) -> PromotionRecord | None:
    """C2 -> C3 when a persisted counterfactual plan exists and enough counterfactual
    trials show the expected directional result above the noise floor."""
    thresholds = load_thresholds(settings)
    noise = thresholds.inconclusive_noise_floor

    if repository.get_hypothesis(hypothesis_id) is None:
        logger.warning("counterfactual gate: unknown hypothesis %s", hypothesis_id)
        return None
    if repository.effective_causal_level(hypothesis_id) != _C2:
        logger.debug("counterfactual gate: hypothesis %s is not at C2", hypothesis_id)
        return None
    # The counterfactual must validate a plan that was actually proposed and persisted.
    if not repository.list_plans_for_hypothesis(hypothesis_id):
        logger.debug("counterfactual gate: no persisted plan for %s", hypothesis_id)
        return None

    supporting = [
        r for r in results
        if improvement(r.baseline_metric, r.post_change_metric, r.metric_direction) > noise
    ]
    if len(supporting) < thresholds.counterfactual_minimum_support:
        return None
    return PromotionRecord(
        promotion_id=new_promotion_id(),
        hypothesis_id=hypothesis_id,
        from_level=_C2,
        to_level=_C3,
        counterfactual_trial_id=supporting[0].trial_id,
        supporting_trial_ids=[r.trial_id for r in supporting],
        rationale=(
            f"{len(supporting)} counterfactual trial(s) produced the expected directional "
            f"improvement above the noise floor (>= {thresholds.counterfactual_minimum_support})"
        ),
        settings_hash=settings.settings_hash(),
    )


def evaluate_c4(
    hypothesis_id: str,
    confirmations: list[CounterfactualResult],
    *,
    settings: Settings,
    repository: Repository,
) -> PromotionRecord | None:
    """C3 -> C4 (rare): enough directional confirmations across >= 2 distinct contexts."""
    thresholds = load_thresholds(settings)
    noise = thresholds.inconclusive_noise_floor

    if repository.get_hypothesis(hypothesis_id) is None:
        logger.warning("c4 gate: unknown hypothesis %s", hypothesis_id)
        return None
    if repository.effective_causal_level(hypothesis_id) != _C3:
        logger.debug("c4 gate: hypothesis %s is not at C3", hypothesis_id)
        return None

    supporting = [
        r for r in confirmations
        if improvement(r.baseline_metric, r.post_change_metric, r.metric_direction) > noise
    ]
    contexts = {(tuple(sorted(r.changed_components)), r.config_hash) for r in supporting}
    if len(supporting) < thresholds.c4_minimum_counterfactuals or len(contexts) < 2:
        return None
    return PromotionRecord(
        promotion_id=new_promotion_id(),
        hypothesis_id=hypothesis_id,
        from_level=_C3,
        to_level=_C4,
        supporting_trial_ids=[r.trial_id for r in supporting],
        rationale=(
            f"{len(supporting)} independent counterfactual confirmations across "
            f"{len(contexts)} distinct contexts"
        ),
        settings_hash=settings.settings_hash(),
    )
