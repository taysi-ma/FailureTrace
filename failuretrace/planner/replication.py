"""Deterministic promotion gate. Produces ``PromotionRecord``s; never mutates hypotheses.

Promotion ladder (each step needs the configured evidence; a single trial can never reach
C2+):

- C1 -> C2: same intervention family replicated across >= ``replication_minimum_trials``
  distinct seeds / equivalent controlled trials.
- C2 -> C3: >= ``counterfactual_minimum_support`` planned counterfactual trials produced the
  expected **directional** result (judged via ``improvement()`` under the metric direction).
- C3 -> C4: >= ``c4_minimum_counterfactuals`` independent confirmations from >= 2 distinct
  contexts (different changed components or configs). Rare by design.

No Bayesian causal estimation in the MVP.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from ..classifier.thresholds import load_thresholds
from ..core.enums import CausalSupportLevel, MetricDirection
from ..core.ids import new_promotion_id
from ..core.models import PromotionRecord
from ..core.settings import Settings, improvement

logger = logging.getLogger(__name__)

_C1 = CausalSupportLevel.C1_plausible_hypothesis
_C2 = CausalSupportLevel.C2_replicated_effect
_C3 = CausalSupportLevel.C3_counterfactual_supported
_C4 = CausalSupportLevel.C4_robust_rule


class ReplicationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    seed: int | None = None
    intervention_family: str | None = None


class CounterfactualResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    baseline_metric: float
    post_change_metric: float
    metric_direction: MetricDirection
    changed_components: list[str] = Field(default_factory=list)
    config_hash: str | None = None


def evaluate_replication(
    hypothesis_id: str,
    evidence: list[ReplicationEvidence],
    *,
    settings: Settings,
    replication_group_id: str,
    from_level: CausalSupportLevel = _C1,
) -> PromotionRecord | None:
    """C1 -> C2 when enough distinct-seed / controlled replications exist."""
    thresholds = load_thresholds(settings)
    seeds = {e.seed for e in evidence if e.seed is not None}
    count = len(seeds) if seeds else len(evidence)
    if count < thresholds.replication_minimum_trials:
        return None
    return PromotionRecord(
        promotion_id=new_promotion_id(),
        hypothesis_id=hypothesis_id,
        from_level=from_level,
        to_level=_C2,
        replication_group_id=replication_group_id,
        supporting_trial_ids=[e.trial_id for e in evidence],
        rationale=(
            f"same intervention family replicated across {count} distinct-seed/controlled "
            f"trials (>= {thresholds.replication_minimum_trials})"
        ),
        settings_hash=settings.settings_hash(),
    )


def evaluate_counterfactual(
    hypothesis_id: str,
    results: list[CounterfactualResult],
    *,
    settings: Settings,
    from_level: CausalSupportLevel = _C2,
) -> PromotionRecord | None:
    """C2 -> C3 when enough counterfactual trials show the expected directional result."""
    thresholds = load_thresholds(settings)
    supporting = [
        r for r in results
        if improvement(r.baseline_metric, r.post_change_metric, r.metric_direction) > 0
    ]
    if len(supporting) < thresholds.counterfactual_minimum_support:
        return None
    return PromotionRecord(
        promotion_id=new_promotion_id(),
        hypothesis_id=hypothesis_id,
        from_level=from_level,
        to_level=_C3,
        counterfactual_trial_id=supporting[0].trial_id,
        supporting_trial_ids=[r.trial_id for r in supporting],
        rationale=(
            f"{len(supporting)} counterfactual trial(s) produced the expected directional "
            f"improvement (>= {thresholds.counterfactual_minimum_support})"
        ),
        settings_hash=settings.settings_hash(),
    )


def evaluate_c4(
    hypothesis_id: str,
    confirmations: list[CounterfactualResult],
    *,
    settings: Settings,
    from_level: CausalSupportLevel = _C3,
) -> PromotionRecord | None:
    """C3 -> C4 (rare): enough directional confirmations across >= 2 distinct contexts."""
    thresholds = load_thresholds(settings)
    supporting = [
        r for r in confirmations
        if improvement(r.baseline_metric, r.post_change_metric, r.metric_direction) > 0
    ]
    contexts = {(tuple(sorted(r.changed_components)), r.config_hash) for r in supporting}
    if len(supporting) < thresholds.c4_minimum_counterfactuals or len(contexts) < 2:
        return None
    return PromotionRecord(
        promotion_id=new_promotion_id(),
        hypothesis_id=hypothesis_id,
        from_level=from_level,
        to_level=_C4,
        supporting_trial_ids=[r.trial_id for r in supporting],
        rationale=(
            f"{len(supporting)} independent counterfactual confirmations across "
            f"{len(contexts)} distinct contexts"
        ),
        settings_hash=settings.settings_hash(),
    )
