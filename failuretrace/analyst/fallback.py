"""Deterministic fallback hypothesis generation.

Converts a :class:`FailureClassification` + :class:`ClassificationContext` into a valid
:class:`FailureHypothesis` with no LLM involvement:

- observations come from the classifier;
- alternative explanations from the classifier's alternative categories + a per-category
  template (kept non-empty for non-deterministic categories, as the model requires);
- missing evidence from absent telemetry fields;
- confidence from the deterministic rubric (the classifier's confidence);
- causal support is C0/C1 only (single trial);
- a hard constraint is set ONLY for ``resource_pressure`` when a configured objective
  VRAM limit is exceeded (condition (b)); the repository still gates persistence.

This path alone must satisfy the whole pipeline — the LLM is strictly additive.
"""

from __future__ import annotations

import logging

from ..classifier.classifier import FailureClassification
from ..classifier.context import ClassificationContext
from ..classifier.thresholds import load_thresholds
from ..core.enums import CausalSupportLevel, FailureCategory, HypothesisSource
from ..core.ids import new_hypothesis_id
from ..core.models import CounterfactualPlanRef, FailureHypothesis, Intervention
from ..core.settings import Settings
from ..telemetry.schema import TelemetryRecord

logger = logging.getLogger(__name__)

C0 = CausalSupportLevel.C0_observation
C1 = CausalSupportLevel.C1_plausible_hypothesis


def _iv(variable: str, action: str, rationale: str, target=None) -> Intervention:
    return Intervention(variable=variable, action=action, target_value=target, rationale=rationale)


# Per-category deterministic templates: hypothesis statement, suggested single-variable
# intervention (aligned with the Phase-4 planner families), counterfactual summary, whether
# a soft penalty is warranted, the causal level, and generic alternative explanations.
_TEMPLATES: dict[FailureCategory, dict] = {
    FailureCategory.divergence: {
        "hypothesis": "Training diverged (NaN/Inf), likely numerical instability from too-high LR or a missing normalization.",
        "intervention": _iv("optimizer.lr", "decrease", "reduce LR to test whether divergence is LR-driven", 0.5),
        "counterfactual": "Halve the LR (hold everything else) and re-run; a stable loss would support the instability hypothesis.",
        "soft_penalty": True, "level": C1,
        "alternatives": ["bf16 numerical edge case", "a bug in a custom kernel/op"],
    },
    FailureCategory.resource_pressure: {
        "hypothesis": "The run hit or approached the VRAM ceiling for this configuration.",
        "intervention": _iv("DEVICE_BATCH_SIZE", "decrease", "reduce per-device batch; preserve effective batch via grad-accum"),
        "counterfactual": "Reduce batch size (or sequence length) with grad-accum to hold effective batch; completion would confirm memory pressure.",
        "soft_penalty": True, "level": C1,
        "alternatives": ["memory fragmentation", "a transient allocation spike unrelated to the change"],
    },
    FailureCategory.runtime_failure: {
        "hypothesis": "A non-OOM runtime error prevented completion, likely a code/config bug introduced by the change.",
        "intervention": _iv("code", "hold", "fix the runtime exception; hold hyperparameters constant, then re-run"),
        "counterfactual": "Fix the exception and re-run unchanged; completion would isolate the bug from the idea.",
        "soft_penalty": False, "level": C1,
        "alternatives": ["environment/dependency issue", "a pre-existing latent bug surfaced by the change"],
    },
    FailureCategory.likely_instability: {
        "hypothesis": "Optimization was unstable (high gradient-norm variability) and the metric regressed.",
        "intervention": _iv("optimizer.lr", "decrease", "halve LR (or raise warmup) to test instability", 0.5),
        "counterfactual": "Hold code + effective batch fixed; halve LR (or increase warmup); recovery would support instability.",
        "soft_penalty": True, "level": C1,
        "alternatives": ["measurement noise / seed variance", "data ordering effects"],
    },
    FailureCategory.likely_undertraining: {
        "hypothesis": "The model was still improving at the fixed budget cutoff; it likely needs more effective training to converge.",
        "intervention": _iv("schedule.horizon", "increase", "increase training budget/steps or reduce model size to converge within budget"),
        "counterfactual": "Hold architecture + optimizer; increase the training budget/steps; improvement would support undertraining.",
        "soft_penalty": True, "level": C1,
        "alternatives": ["learning-rate schedule mistuned", "the change is simply neutral"],
    },
    FailureCategory.possible_overfitting: {
        "hypothesis": "A growing train/val gap with a val regression suggests overfitting.",
        "intervention": _iv("WEIGHT_DECAY", "increase", "increase one regularizer to test overfitting"),
        "counterfactual": "Hold architecture; increase exactly one regularizer; val recovery would support overfitting.",
        "soft_penalty": True, "level": C1,
        "alternatives": ["distribution shift in the validation shard", "insufficient data for the model size"],
    },
    FailureCategory.possible_over_regularization: {
        "hypothesis": "Stronger regularization worsened both train and val — likely over-regularization.",
        "intervention": _iv("WEIGHT_DECAY", "decrease", "reduce the added regularization to test over-regularization"),
        "counterfactual": "Halve the added regularization; recovery of both train and val would confirm over-regularization.",
        "soft_penalty": True, "level": C1,
        "alternatives": ["regularization interacts with the LR schedule", "measurement noise"],
    },
    FailureCategory.invalid_comparison: {
        "hypothesis": "The baseline/post comparison is not valid (missing baseline, or metric/seed/protocol mismatch).",
        "intervention": _iv("evaluation.protocol", "hold", "re-establish a matched baseline, seed, metric, and eval protocol"),
        "counterfactual": "Re-run with a matched baseline, seed, metric, and eval protocol; a valid delta would enable a verdict.",
        "soft_penalty": False, "level": C0,
        "alternatives": ["a logging/recording error rather than a real protocol change", "a transient eval-set difference"],
    },
    FailureCategory.inconclusive: {
        "hypothesis": "No signal above the configured noise floor; the change appears neutral.",
        "intervention": _iv("experiment", "hold", "replicate across seeds to detect a small true effect, if any"),
        "counterfactual": "Replicate across seeds; a consistent directional delta would reveal a small true effect.",
        "soft_penalty": False, "level": C0,
        "alternatives": ["a real but tiny effect below current sensitivity", "seed variance"],
    },
    FailureCategory.unknown: {
        "hypothesis": "No rule matched the available evidence; the cause is undetermined.",
        "intervention": _iv("telemetry", "hold", "collect gradient/loss telemetry, then re-run to enable classification"),
        "counterfactual": "Collect gradient-norm and loss telemetry and re-run to enable a classification.",
        "soft_penalty": False, "level": C0,
        "alternatives": ["insufficient telemetry to classify", "a failure mode not yet modeled"],
    },
}

# Informative telemetry fields used to score evidence quality / missing evidence.
_INFORMATIVE_FIELDS: dict[str, str] = {
    "val_metric": "validation metric",
    "train_loss_start": "initial train loss",
    "train_loss_end": "final train loss",
    "gradient_norm_cv": "gradient-norm variability",
    "gradient_norm_mean": "gradient-norm mean",
    "train_val_gap": "train/val gap",
    "learning_rate_history": "LR schedule",
    "peak_vram_gb": "peak VRAM",
    "runtime_seconds": "runtime",
}


def _evidence_quality(tel: TelemetryRecord) -> float:
    present = sum(1 for f in _INFORMATIVE_FIELDS if getattr(tel, f) is not None)
    return round(present / len(_INFORMATIVE_FIELDS), 4)


def _missing_evidence(tel: TelemetryRecord) -> list[str]:
    return [f"missing {label}" for field, label in _INFORMATIVE_FIELDS.items() if getattr(tel, field) is None]


def _telemetry_evidence(ctx: ClassificationContext) -> list[str]:
    tel = ctx.telemetry
    evidence: list[str] = []
    if ctx.baseline_metric is not None and ctx.post_change_metric is not None:
        evidence.append(
            f"baseline={ctx.baseline_metric}, post={ctx.post_change_metric} (direction={ctx.metric_direction})"
        )
    if tel.gradient_norm_cv is not None:
        evidence.append(f"gradient_norm_cv={tel.gradient_norm_cv:.3f}")
    if tel.peak_vram_gb is not None:
        evidence.append(f"peak_vram_gb={tel.peak_vram_gb:.2f}")
    if ctx.exception_type:
        evidence.append(f"exception_type={ctx.exception_type}")
    return evidence


def _hard_constraint(category: FailureCategory, tel: TelemetryRecord, limit_gb: float | None) -> bool:
    # Only an objectively-exceeded resource limit justifies a hard constraint from a single trial.
    if category == FailureCategory.resource_pressure and limit_gb is not None:
        if tel.peak_vram_gb is not None and tel.peak_vram_gb >= limit_gb:
            return True
    return False


def build_fallback(
    classification: FailureClassification,
    ctx: ClassificationContext,
    *,
    trial_id: str,
    settings: Settings,
    source: HypothesisSource = HypothesisSource.rule_based,
) -> FailureHypothesis:
    """Build a deterministic, valid :class:`FailureHypothesis` (no LLM)."""
    template = _TEMPLATES[classification.category]
    thresholds = load_thresholds(settings)
    tel = ctx.telemetry

    alternatives = [f"could instead be {cat.value}" for cat in classification.alternative_categories]
    alternatives += list(template["alternatives"])

    evidence = [f"triggered rule: {rule}" for rule in classification.triggered_rules]
    evidence += _telemetry_evidence(ctx)

    hypothesis = FailureHypothesis(
        hypothesis_id=new_hypothesis_id(),
        trial_id=trial_id,
        source=source,
        category=classification.category,
        observations=list(classification.observations) or ["no observations recorded"],
        evidence=evidence or ["no telemetry evidence available"],
        hypotheses=[template["hypothesis"]],
        alternative_explanations=alternatives,
        missing_evidence=_missing_evidence(tel),
        hypothesis_confidence=classification.confidence,
        evidence_quality=_evidence_quality(tel),
        suggested_intervention=template["intervention"],
        proposed_counterfactual_trial=CounterfactualPlanRef(summary=template["counterfactual"]),
        should_apply_soft_penalty=template["soft_penalty"],
        should_apply_hard_constraint=_hard_constraint(
            classification.category, tel, thresholds.resource_vram_limit_gb
        ),
        causal_support_level=template["level"],
        settings_hash=classification.settings_hash,
    )
    logger.debug(
        "built fallback hypothesis %s (category=%s, source=%s)",
        hypothesis.hypothesis_id, hypothesis.category, source,
    )
    return hypothesis
