"""Persisted Pydantic models with model-level enforcement of the epistemic rules.

Records are immutable after construction (``frozen=True``) — changes are expressed as
new linked records, never mutations (see spec Cross-Cutting Invariant #2).

Epistemic guardrails enforced *on the model* (reload-safe, based only on persisted
fields):

- confidence / quality bounded to [0, 1];
- a freshly built ``FailureHypothesis`` may assert only C0/C1 (C2+ requires a
  ``PromotionRecord``);
- ``alternative_explanations`` must be non-empty unless the category is deterministic;
- ``inconclusive`` evidence can never set a hard constraint;
- ``should_apply_hard_constraint`` may only be *set* for objectively-deterministic
  categories — a single noisy performance regression can never yield a hard constraint.

The *conditional* hard-constraint justifications that need cross-trial/telemetry/
promotion context — (a) deterministic AND repeated, (b) an objective resource limit
exceeded, (c) effective level >= C2 — are enforced at write time by the repository
(the only write path), which refuses to persist a violating record.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    CausalSupportLevel,
    FailureCategory,
    HypothesisSource,
    LinkType,
    MetricDirection,
    TrialStatus,
)

# Categories whose assignment already implies objective evidence (OOM / NaN-Inf / an
# uncaught runtime exception). These are exempt from the alternative-explanations
# requirement and are the only categories for which a hard constraint may be *set* on a
# freshly created record (the repository still gates on repeated/objective/C2 evidence).
DETERMINISTIC_CATEGORIES: frozenset[FailureCategory] = frozenset(
    {
        FailureCategory.divergence,
        FailureCategory.resource_pressure,
        FailureCategory.runtime_failure,
    }
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrialRecord(BaseModel):
    """An immutable record of a single experiment trial (ingested at a terminal state)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_id: str
    parent_trial_id: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    git_commit: str | None = None
    config_hash: str | None = None
    seed: int | None = None
    status: TrialStatus

    metric_name: str
    metric_direction: MetricDirection
    baseline_metric: float | None = None
    post_change_metric: float | None = None
    # Raw difference ``post - baseline``; interpretation always via improvement().
    metric_delta: float | None = None

    runtime_seconds: float | None = None
    # Canonical peak VRAM; copied from telemetry at ingestion when present.
    peak_vram_gb: float | None = None
    throughput: float | None = None

    exception_type: str | None = None
    exception_message: str | None = None

    code_diff: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    changed_components: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            telemetry = data.get("telemetry")
            if (
                data.get("peak_vram_gb") is None
                and isinstance(telemetry, dict)
                and telemetry.get("peak_vram_gb") is not None
            ):
                data["peak_vram_gb"] = telemetry["peak_vram_gb"]

            baseline = data.get("baseline_metric")
            post = data.get("post_change_metric")
            if data.get("metric_delta") is None and baseline is not None and post is not None:
                data["metric_delta"] = post - baseline
        return data


class Intervention(BaseModel):
    """A single proposed change to one variable."""

    model_config = ConfigDict(extra="forbid")

    variable: str  # e.g. "optimizer.lr"
    action: Literal["decrease", "increase", "set", "hold"]
    target_value: float | str | None = None
    rationale: str


class CounterfactualPlanRef(BaseModel):
    """A reference to a (possibly not-yet-persisted) counterfactual validation plan."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str | None = None
    summary: str


class FailureHypothesis(BaseModel):
    """A falsifiable, uncertainty-aware hypothesis about why a trial failed.

    Claim strength lives *only* in ``causal_support_level``; belief strength in
    ``hypothesis_confidence``. There is deliberately no ``causal_confidence`` field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    trial_id: str
    source: HypothesisSource
    category: FailureCategory

    observations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    hypothesis_confidence: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    # The LLM's *stated* belief, recorded for provenance only. It never replaces the
    # deterministic rubric value in ``hypothesis_confidence`` (Cross-Cutting Invariant 5:
    # confidence comes from the rubric, never ad-hoc). ``None`` on rule-based hypotheses.
    llm_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    suggested_intervention: Intervention
    proposed_counterfactual_trial: CounterfactualPlanRef

    should_apply_soft_penalty: bool = False
    should_apply_hard_constraint: bool = False

    causal_support_level: CausalSupportLevel
    settings_hash: str

    @model_validator(mode="after")
    def _epistemic_rules(self) -> "FailureHypothesis":
        # C2+ can only be asserted via a PromotionRecord, never on a fresh record.
        if self.causal_support_level.at_least(CausalSupportLevel.C2_replicated_effect):
            raise ValueError(
                "causal_support_level must be C0 or C1 at creation; "
                "C2+ is asserted only by PromotionRecord"
            )

        # Non-deterministic categories must offer alternative explanations.
        if self.category not in DETERMINISTIC_CATEGORIES and not self.alternative_explanations:
            raise ValueError(
                f"alternative_explanations must be non-empty for non-deterministic "
                f"category '{self.category}'"
            )

        if self.should_apply_hard_constraint:
            # Inconclusive evidence never yields a hard restriction.
            if self.category == FailureCategory.inconclusive:
                raise ValueError("inconclusive evidence cannot set a hard constraint")
            # A single noisy performance regression can never be a hard constraint:
            # only objectively-deterministic categories may even *set* the flag; the
            # repository still verifies repeated/objective/C2 justification at write time.
            if self.category not in DETERMINISTIC_CATEGORIES:
                raise ValueError(
                    f"hard constraint not permitted for category '{self.category}' on a "
                    f"single record (requires deterministic repeated failure, an objective "
                    f"resource limit, or C2+ via promotion)"
                )
        return self


class PromotionRecord(BaseModel):
    """Append-only upgrade of a hypothesis's causal support level.

    A hypothesis's *effective* level is its original level overridden by the highest
    valid PromotionRecord. Hypothesis records are never mutated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    promotion_id: str
    hypothesis_id: str
    from_level: CausalSupportLevel
    to_level: CausalSupportLevel
    timestamp: datetime = Field(default_factory=_utcnow)
    replication_group_id: str | None = None
    counterfactual_trial_id: str | None = None
    supporting_trial_ids: list[str] = Field(default_factory=list)
    rationale: str
    settings_hash: str

    @model_validator(mode="after")
    def _monotonic_promotion(self) -> "PromotionRecord":
        if self.to_level.rank <= self.from_level.rank:
            raise ValueError(
                f"promotion must increase causal support level "
                f"({self.from_level} -> {self.to_level})"
            )
        if not self.to_level.at_least(CausalSupportLevel.C2_replicated_effect):
            raise ValueError("promotions may only assert C2 or higher")
        return self


class LinkRecord(BaseModel):
    """Append-only evidence link (replication / validation / counterfactual)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    link_id: str
    link_type: LinkType
    hypothesis_id: str | None = None
    trial_id: str | None = None
    source_trial_id: str | None = None
    counterfactual_trial_id: str | None = None
    replication_group_id: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    settings_hash: str | None = None
    note: str | None = None


class CounterfactualPlan(BaseModel):
    """A deterministic, controlled validation experiment (append-only; never executed).

    By default exactly one primary variable is intervened on. A coupled stabilization
    variable is permitted ONLY when the hypothesis explicitly concerns the interaction
    of the two — in which case ``interaction_rationale`` must state why both are
    necessary and which interaction is tested (the validator rejects a coupled plan
    without it). Expected outcomes are stated direction-aware (via ``improvement()``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    hypothesis_id: str
    primary_intervention_variable: str
    optional_coupled_stabilization_variable: str | None = None
    control_variables: list[str] = Field(default_factory=list)
    treatment_variables: list[str] = Field(default_factory=list)
    held_constant_variables: list[str] = Field(default_factory=list)
    expected_outcome_if_hypothesis_true: str
    expected_outcome_if_hypothesis_false: str
    interaction_rationale: str | None = None
    settings_hash: str

    @model_validator(mode="after")
    def _coupled_requires_rationale(self) -> "CounterfactualPlan":
        if self.optional_coupled_stabilization_variable is not None:
            if not (self.interaction_rationale and self.interaction_rationale.strip()):
                raise ValueError(
                    "a coupled plan (two variables) requires a non-empty interaction_rationale"
                )
        return self
