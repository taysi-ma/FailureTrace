"""Deterministic controlled effect-size estimator.

Reads a hypothesis's counterfactual (controlled) trials — the same evidence the C2->C3
gate uses — and summarizes the *magnitude* of the effect with a closed-form interval.
No priors, no sampling, no confounder adjustment: identification is by the controlled
design (the counterfactual plan holds everything else constant); estimation is plain
summary statistics on the direction-aware deltas. Same records in => byte-identical
numbers out.

Preconditions (else returns ``None``): the hypothesis exists, its *effective* level is
>= C3 (so controlled validation actually happened), and it has >= 1 counterfactual-linked
trial with both metrics present.
"""

from __future__ import annotations

import logging
from math import sqrt
from statistics import fmean, stdev

from pydantic import BaseModel, ConfigDict, Field

from ..core.enums import CausalSupportLevel
from ..core.ids import new_estimate_id
from ..core.models import EffectEstimate
from ..core.settings import Settings, improvement
from ..planner.replication import _counterfactual_results
from ..store.repository import Repository

logger = logging.getLogger(__name__)

_C3 = CausalSupportLevel.C3_counterfactual_supported
_DECIMALS = 6


class EstimationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    confidence_level: float = 0.90
    min_counterfactuals_for_interval: int = 2
    small_sample_widening: float = 1.0
    z_by_level: dict[str, float] = Field(default_factory=dict)

    def z(self) -> float:
        """Standard-normal quantile for ``confidence_level`` (from the config table)."""
        key = f"{self.confidence_level:.2f}"
        return self.z_by_level.get(key, self.z_by_level.get("0.90", 1.6449))


def load_estimation_config(settings: Settings) -> EstimationConfig:
    return EstimationConfig(**settings.section("estimation"))


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, _DECIMALS)


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def estimate_effect(
    hypothesis_id: str,
    *,
    settings: Settings,
    repository: Repository,
) -> EffectEstimate | None:
    """Compute a controlled effect-size estimate, or ``None`` if preconditions fail."""
    cfg = load_estimation_config(settings)
    if not cfg.enabled:
        return None

    hyp = repository.get_hypothesis(hypothesis_id)
    if hyp is None:
        return None
    level = repository.effective_causal_level(hypothesis_id)
    if level is None or not level.at_least(_C3):
        logger.debug("effect estimate: hypothesis %s is below C3", hypothesis_id)
        return None

    results = _counterfactual_results(repository, hypothesis_id)
    if not results:
        return None

    deltas = [
        improvement(r.baseline_metric, r.post_change_metric, r.metric_direction)
        for r in results
    ]
    n = len(deltas)
    absolute = fmean(deltas)

    # Relative effect: mean of per-trial improvement / |baseline| (skip near-zero baselines).
    relatives = [
        improvement(r.baseline_metric, r.post_change_metric, r.metric_direction) / abs(r.baseline_metric)
        for r in results
        if abs(r.baseline_metric) > 1e-12
    ]
    relative = fmean(relatives) if relatives else None

    dispersion = stdev(deltas) if n >= 2 else None
    standardized = (absolute / dispersion) if dispersion else None

    ci_low = ci_high = None
    if n >= cfg.min_counterfactuals_for_interval and dispersion:
        se = (dispersion / sqrt(n)) * cfg.small_sample_widening
        z = cfg.z()
        ci_low, ci_high = absolute - z * se, absolute + z * se

    sign = _sign(absolute)
    consistency = 1.0 if sign == 0 else sum(1 for d in deltas if _sign(d) == sign) / n

    # Attach the highest counterfactual/robust promotion for provenance.
    promotion_id = None
    for promo in repository.list_promotions_for_hypothesis(hypothesis_id):
        if promo.to_level.at_least(_C3):
            promotion_id = promo.promotion_id

    return EffectEstimate(
        estimate_id=new_estimate_id(),
        hypothesis_id=hypothesis_id,
        promotion_id=promotion_id,
        metric_name=settings.metric.name,
        metric_direction=results[0].metric_direction,
        n_counterfactuals=n,
        absolute_effect=_round(absolute),
        relative_effect=_round(relative),
        standardized_effect=_round(standardized),
        dispersion=_round(dispersion),
        ci_low=_round(ci_low),
        ci_high=_round(ci_high),
        range_low=_round(min(deltas)),
        range_high=_round(max(deltas)),
        confidence_level=cfg.confidence_level,
        consistency=round(consistency, 4),
        supporting_trial_ids=[r.trial_id for r in results],
        method="controlled_delta_v1",
        settings_hash=settings.settings_hash(),
    )


def _fingerprint(supporting_trial_ids: list[str], settings_hash: str, method: str) -> tuple:
    return (tuple(sorted(supporting_trial_ids)), settings_hash, method)


def estimate_effects(repository: Repository, settings: Settings) -> list[EffectEstimate]:
    """Scan effective-C3+ hypotheses and persist a fresh estimate for each whose evidence
    has changed. Idempotent: an estimate over the same supporting set + settings + method is
    not re-written, so re-running produces no duplicates."""
    cfg = load_estimation_config(settings)
    if not cfg.enabled:
        return []

    written: list[EffectEstimate] = []
    for hyp in repository.list_hypotheses():
        level = repository.effective_causal_level(hyp.hypothesis_id)
        if level is None or not level.at_least(_C3):
            continue
        estimate = estimate_effect(hyp.hypothesis_id, settings=settings, repository=repository)
        if estimate is None:
            continue
        existing = repository.list_effect_estimates_for_hypothesis(hyp.hypothesis_id)
        target = _fingerprint(estimate.supporting_trial_ids, estimate.settings_hash, estimate.method)
        if any(
            _fingerprint(e.supporting_trial_ids, e.settings_hash, e.method) == target
            for e in existing
        ):
            continue
        repository.save_effect_estimate(estimate)
        written.append(estimate)
    logger.info("effect-size estimator wrote %d new estimate(s)", len(written))
    return written
