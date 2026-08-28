"""Deterministic promotion gate. Produces ``PromotionRecord``s; never mutates hypotheses.

Promotion ladder (each step needs the configured evidence, verified against the *store* —
a single trial can never reach C2+, and fabricated or mismatched evidence is rejected):

- C1 -> C2: the same intervention family (matched by the source trial's fingerprint)
  reproduced across >= ``replication_minimum_trials`` distinct seeds / equivalent
  controlled trials, all pointing the *same* metric direction and clearing the noise floor.
- C2 -> C3: a **persisted** counterfactual plan exists for the hypothesis AND
  >= ``counterfactual_minimum_support`` counterfactual trials CONFIRMED the plan's
  prediction. What "confirmed" means is per-category and configured in
  ``counterfactual.success_criterion``: ``metric_improvement`` (the default) demands a
  direction-aware ``improvement()`` above the noise floor, while ``completion`` demands
  only that the trial ran to completion — the correct test for ``resource_pressure``,
  whose prediction is "the run stops failing", not "the metric improves".
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
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from ..classifier.thresholds import load_thresholds
from ..core.enums import CausalSupportLevel, FailureCategory, LinkType, MetricDirection, TrialStatus
from ..core.ids import new_link_id, new_promotion_id, new_replication_group_id
from ..core.models import LinkRecord, PromotionRecord, TrialRecord
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
    # Terminal status of the counterfactual trial, needed by the ``completion`` criterion.
    status: TrialStatus | None = None


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


def _replication_unit(trial: TrialRecord, seed: int | None) -> tuple:
    """A distinct replication unit: the (seed, git_commit) pair, so independent commits
    count separately (even under a pinned seed) while identical re-runs collapse to one.
    Falls back to the trial id only when neither seed nor commit is known."""
    resolved_seed = seed if seed is not None else trial.seed
    if resolved_seed is not None or trial.git_commit:
        return (resolved_seed, trial.git_commit)
    return ("trial", trial.trial_id)


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

    qualifying: list[str] = []
    units: set[tuple] = set()
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
        qualifying.append(trial.trial_id)
        units.add(_replication_unit(trial, ev.seed))

    # A replication *unit* is a distinct (seed, commit) pair: independent runs on different
    # commits count separately even under a pinned seed (autoresearch fixes seed 42), while
    # deterministic re-runs of the same commit+seed collapse to one and cannot inflate count.
    count = len(units)
    if count < thresholds.replication_minimum_trials:
        return None

    return PromotionRecord(
        promotion_id=new_promotion_id(),
        hypothesis_id=hypothesis_id,
        from_level=_C1,
        to_level=_C2,
        replication_group_id=replication_group_id,
        supporting_trial_ids=qualifying,
        rationale=(
            f"same intervention family replicated across {count} distinct (seed, commit) "
            f"units (>= {thresholds.replication_minimum_trials}), consistent direction"
        ),
        settings_hash=settings.settings_hash(),
    )


def promote_replications(repository: Repository, settings: Settings) -> list[PromotionRecord]:
    """Scan persisted C1 hypotheses, group them by (category, intervention fingerprint),
    and promote each group that clears the replication gate to C2 — persisting the
    promotion and an append-only replication :class:`LinkRecord` per supporting trial.

    This is the deterministic driver behind ``failuretrace gate``. It is the only place a
    replication promotion is minted in the normal flow; ``evaluate_replication`` and the
    repository write-gate still enforce the evidence requirements.
    """
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for hyp in repository.list_hypotheses():
        trial = repository.get_trial(hyp.trial_id)
        if trial is None:
            continue
        level = repository.effective_causal_level(hyp.hypothesis_id)
        key = (hyp.category, _intervention_fingerprint(trial))
        groups[key].append((hyp, trial, level))

    minimum = load_thresholds(settings).replication_minimum_trials
    promotions: list[PromotionRecord] = []
    for members in groups.values():
        # Idempotent: a group already represented by a replication promotion is skipped,
        # so re-running the gate does not keep promoting the C1 stragglers of one effect.
        if any(level is not None and level.at_least(_C2) for _, _, level in members):
            continue
        c1_members = [(h, t) for h, t, level in members if level == _C1]
        if len(c1_members) < minimum:
            continue
        representative = c1_members[0][0]
        group_id = new_replication_group_id()
        evidence = [ReplicationEvidence(trial_id=t.trial_id, seed=t.seed) for _, t in c1_members]
        promotion = evaluate_replication(
            representative.hypothesis_id, evidence,
            settings=settings, repository=repository, replication_group_id=group_id,
        )
        if promotion is None:
            continue
        repository.save_promotion(promotion)
        # Explicit, append-only replication links (spec §1.3): each supporting trial ->
        # the promoted hypothesis, tagged with the replication group.
        for trial_id in promotion.supporting_trial_ids:
            repository.save_link(LinkRecord(
                link_id=new_link_id(),
                link_type=LinkType.replication,
                hypothesis_id=representative.hypothesis_id,
                trial_id=trial_id,
                replication_group_id=group_id,
                settings_hash=settings.settings_hash(),
                note="replication support for C1->C2 promotion",
            ))
        promotions.append(promotion)
    logger.info("replication gate promoted %d hypothesis group(s) to C2", len(promotions))
    return promotions


# A counterfactual trial "ran to completion" when it reached a terminal status that
# implies the run itself finished. ``failed_oom`` / ``failed_runtime`` are excluded: those
# are exactly the outcomes a resource_pressure intervention predicts will stop happening.
_COMPLETED_STATUSES = frozenset({
    TrialStatus.completed, TrialStatus.promoted, TrialStatus.rejected,
})

_METRIC_IMPROVEMENT = "metric_improvement"
_COMPLETION = "completion"
_VALID_CRITERIA = frozenset({_METRIC_IMPROVEMENT, _COMPLETION})


def success_criterion_for(category: FailureCategory, settings: Settings) -> str:
    """The configured C2->C3 confirmation rule for *category*.

    Unknown categories fall back to ``default``; an unrecognised configured value falls
    back to ``metric_improvement`` with a warning rather than silently promoting.
    """
    section = settings.section("counterfactual").get("success_criterion", {}) or {}
    criterion = section.get(category.value, section.get("default", _METRIC_IMPROVEMENT))
    if criterion not in _VALID_CRITERIA:
        logger.warning(
            "unknown counterfactual success_criterion %r for %s; falling back to %s",
            criterion, category.value, _METRIC_IMPROVEMENT,
        )
        return _METRIC_IMPROVEMENT
    return criterion


def _confirms(result: CounterfactualResult, criterion: str, noise: float) -> bool:
    """Did this counterfactual trial confirm the plan's prediction?"""
    if criterion == _COMPLETION:
        return result.status in _COMPLETED_STATUSES
    return improvement(
        result.baseline_metric, result.post_change_metric, result.metric_direction
    ) > noise


def evaluate_counterfactual(
    hypothesis_id: str,
    results: list[CounterfactualResult],
    *,
    settings: Settings,
    repository: Repository,
) -> PromotionRecord | None:
    """C2 -> C3 when a persisted counterfactual plan exists and enough counterfactual
    trials confirmed it under the category's configured success criterion."""
    thresholds = load_thresholds(settings)
    noise = thresholds.inconclusive_noise_floor

    hypothesis = repository.get_hypothesis(hypothesis_id)
    if hypothesis is None:
        logger.warning("counterfactual gate: unknown hypothesis %s", hypothesis_id)
        return None
    if repository.effective_causal_level(hypothesis_id) != _C2:
        logger.debug("counterfactual gate: hypothesis %s is not at C2", hypothesis_id)
        return None
    # The counterfactual must validate a plan that was actually proposed and persisted.
    if not repository.list_plans_for_hypothesis(hypothesis_id):
        logger.debug("counterfactual gate: no persisted plan for %s", hypothesis_id)
        return None

    criterion = success_criterion_for(hypothesis.category, settings)
    supporting = [r for r in results if _confirms(r, criterion, noise)]
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
            f"{len(supporting)} counterfactual trial(s) confirmed the plan under the "
            f"{criterion!r} criterion "
            f"(>= {thresholds.counterfactual_minimum_support} required)"
        ),
        settings_hash=settings.settings_hash(),
    )


def link_counterfactual_trial(
    repository: Repository,
    settings: Settings,
    *,
    hypothesis_id: str,
    counterfactual_trial_id: str,
) -> LinkRecord:
    """Record that a persisted trial was run to validate a hypothesis's counterfactual plan.

    This append-only ``counterfactual`` link is what :func:`promote_counterfactuals` and
    :func:`promote_c4` later read to accumulate validation evidence. Both the hypothesis and
    the trial must already exist (enforced by the links table foreign keys)."""
    return repository.save_link(LinkRecord(
        link_id=new_link_id(),
        link_type=LinkType.counterfactual,
        hypothesis_id=hypothesis_id,
        trial_id=counterfactual_trial_id,           # FK-enforced existence
        counterfactual_trial_id=counterfactual_trial_id,
        settings_hash=settings.settings_hash(),
        note="counterfactual validation trial for a proposed plan",
    ))


def _counterfactual_results(repository: Repository, hypothesis_id: str) -> list[CounterfactualResult]:
    """Build ``CounterfactualResult``s from every counterfactual-linked trial of a hypothesis."""
    results: list[CounterfactualResult] = []
    for link in repository.list_links_for_hypothesis(hypothesis_id):
        if link.link_type != LinkType.counterfactual or not link.counterfactual_trial_id:
            continue
        trial = repository.get_trial(link.counterfactual_trial_id)
        if trial is None or trial.baseline_metric is None or trial.post_change_metric is None:
            continue
        results.append(CounterfactualResult(
            trial_id=trial.trial_id,
            baseline_metric=trial.baseline_metric,
            post_change_metric=trial.post_change_metric,
            metric_direction=trial.metric_direction,
            changed_components=trial.changed_components,
            config_hash=trial.config_hash,
            status=trial.status,
        ))
    return results


def promote_counterfactuals(repository: Repository, settings: Settings) -> list[PromotionRecord]:
    """Scan effective-C2 hypotheses and promote to C3 those whose linked counterfactual
    trials produced the expected directional result (a persisted plan is required by the
    gate). Persists the promotion and a ``validation`` link. Idempotent: a hypothesis past
    C2 is skipped."""
    promotions: list[PromotionRecord] = []
    for hyp in repository.list_hypotheses():
        if repository.effective_causal_level(hyp.hypothesis_id) != _C2:
            continue
        results = _counterfactual_results(repository, hyp.hypothesis_id)
        if not results:
            continue
        promotion = evaluate_counterfactual(
            hyp.hypothesis_id, results, settings=settings, repository=repository
        )
        if promotion is None:
            continue
        repository.save_promotion(promotion)
        repository.save_link(LinkRecord(
            link_id=new_link_id(),
            link_type=LinkType.validation,
            hypothesis_id=hyp.hypothesis_id,
            trial_id=promotion.counterfactual_trial_id,
            counterfactual_trial_id=promotion.counterfactual_trial_id,
            settings_hash=settings.settings_hash(),
            note="counterfactual validation supporting C2->C3",
        ))
        promotions.append(promotion)
    logger.info("counterfactual gate promoted %d hypothesis(es) to C3", len(promotions))
    return promotions


def promote_c4(repository: Repository, settings: Settings) -> list[PromotionRecord]:
    """Scan effective-C3 hypotheses and promote to C4 those with enough counterfactual
    confirmations across >= 2 distinct contexts. Rare by design; idempotent."""
    promotions: list[PromotionRecord] = []
    for hyp in repository.list_hypotheses():
        if repository.effective_causal_level(hyp.hypothesis_id) != _C3:
            continue
        confirmations = _counterfactual_results(repository, hyp.hypothesis_id)
        promotion = evaluate_c4(
            hyp.hypothesis_id, confirmations, settings=settings, repository=repository
        )
        if promotion is None:
            continue
        repository.save_promotion(promotion)
        promotions.append(promotion)
    logger.info("c4 gate promoted %d hypothesis(es) to C4", len(promotions))
    return promotions


def advance_promotions(repository: Repository, settings: Settings) -> dict[str, list]:
    """Run the full promotion ladder in order — replication (C1->C2), counterfactual
    (C2->C3), then C4 (C3->C4) — so a hypothesis with sufficient accumulated evidence can
    climb multiple rungs in one pass, then estimate controlled effect sizes for any
    hypothesis now at C3+.

    Each step is individually idempotent. The ``effects`` value holds any newly-written
    ``EffectEstimate``s (empty when estimation is disabled or nothing changed) — it is the
    only non-``PromotionRecord`` value, so callers that count promotions must select the
    ``replication``/``counterfactual``/``c4`` keys explicitly."""
    from ..estimation.effect import estimate_effects  # local import: avoid an import cycle

    return {
        "replication": promote_replications(repository, settings),
        "counterfactual": promote_counterfactuals(repository, settings),
        "c4": promote_c4(repository, settings),
        "effects": estimate_effects(repository, settings),
    }


def evaluate_c4(
    hypothesis_id: str,
    confirmations: list[CounterfactualResult],
    *,
    settings: Settings,
    repository: Repository,
) -> PromotionRecord | None:
    """C3 -> C4 (rare): enough confirmations across >= 2 distinct contexts.

    Uses the same per-category criterion as the C2->C3 gate, so a category judged by
    ``completion`` is not promoted to C3 and then stranded there by a different rule."""
    thresholds = load_thresholds(settings)
    noise = thresholds.inconclusive_noise_floor

    hypothesis = repository.get_hypothesis(hypothesis_id)
    if hypothesis is None:
        logger.warning("c4 gate: unknown hypothesis %s", hypothesis_id)
        return None
    if repository.effective_causal_level(hypothesis_id) != _C3:
        logger.debug("c4 gate: hypothesis %s is not at C3", hypothesis_id)
        return None

    criterion = success_criterion_for(hypothesis.category, settings)
    supporting = [r for r in confirmations if _confirms(r, criterion, noise)]
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
