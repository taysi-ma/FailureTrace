"""Deterministic, explainable classification rules.

Each rule is a small named function returning a :class:`RuleResult` (or ``None``).
Thresholds are injected from Settings — never hard-coded. Every rule declares a
confidence *tier* and the *required evidence fields* used to cap confidence by
completeness (see :mod:`failuretrace.classifier.classifier` for the rubric).

Direction awareness: a "regression" means ``improvement(baseline, post, direction) < 0``
via the single canonical helper — never "larger/smaller is worse" assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.enums import FailureCategory
from ..core.settings import improvement
from .context import ClassificationContext, field_value
from .thresholds import Thresholds


@dataclass(frozen=True)
class RuleResult:
    category: FailureCategory
    tier: str
    observation: str
    triggered_rule: str
    required_fields: tuple[str, ...] = ()


def _is_oom(exc_type: str | None, exc_msg: str | None) -> bool:
    blob = f"{exc_type or ''} {exc_msg or ''}".lower()
    return (
        "outofmemory" in blob.replace(" ", "")
        or "out of memory" in blob
        or "oom" in blob.split()
    )


def _improvement(ctx: ClassificationContext) -> float | None:
    if ctx.baseline_metric is None or ctx.post_change_metric is None:
        return None
    return improvement(ctx.baseline_metric, ctx.post_change_metric, ctx.metric_direction)


def _regressed(ctx: ClassificationContext) -> bool | None:
    imp = _improvement(ctx)
    return None if imp is None else imp < 0


def _not_improved(ctx: ClassificationContext) -> bool | None:
    imp = _improvement(ctx)
    return None if imp is None else imp <= 0


def rule_divergence(ctx: ClassificationContext, thr: Thresholds) -> RuleResult | None:
    nan = field_value(ctx, "nan_detected")
    inf = field_value(ctx, "inf_detected")
    if nan is True or inf is True:
        return RuleResult(
            FailureCategory.divergence,
            "deterministic",
            f"NaN/Inf detected during training (nan_detected={nan}, inf_detected={inf})",
            "divergence_nan_inf",
        )
    return None


def rule_resource_pressure(ctx: ClassificationContext, thr: Thresholds) -> RuleResult | None:
    if _is_oom(ctx.exception_type, ctx.exception_message):
        return RuleResult(
            FailureCategory.resource_pressure,
            "deterministic",
            f"out-of-memory during run ({ctx.exception_type or 'OOM'})",
            "resource_pressure_oom",
        )
    ratio = field_value(ctx, "gpu_memory_ratio")
    if ratio is not None and ratio >= thr.gpu_memory_ratio_resource_pressure:
        return RuleResult(
            FailureCategory.resource_pressure,
            "deterministic",
            f"gpu_memory_ratio {ratio:.3f} >= {thr.gpu_memory_ratio_resource_pressure}",
            "resource_pressure_gpu_ratio",
            ("gpu_memory_ratio",),
        )
    return None


def rule_runtime_failure(ctx: ClassificationContext, thr: Thresholds) -> RuleResult | None:
    if ctx.exception_type and not _is_oom(ctx.exception_type, ctx.exception_message):
        detail = f" — {ctx.exception_message}" if ctx.exception_message else ""
        return RuleResult(
            FailureCategory.runtime_failure,
            "deterministic",
            f"runtime exception: {ctx.exception_type}{detail}",
            "runtime_failure_exception",
        )
    return None


def rule_invalid_comparison(ctx: ClassificationContext, thr: Thresholds) -> RuleResult | None:
    reasons: list[str] = []
    if ctx.baseline_metric is None:
        reasons.append("baseline metric missing")
    if (
        ctx.baseline_metric_name
        and ctx.post_metric_name
        and ctx.baseline_metric_name != ctx.post_metric_name
    ):
        reasons.append(
            f"metric_name differs ({ctx.baseline_metric_name} vs {ctx.post_metric_name})"
        )
    if (
        ctx.requires_matched_seeds
        and ctx.baseline_seed is not None
        and ctx.post_seed is not None
        and ctx.baseline_seed != ctx.post_seed
    ):
        reasons.append(f"seed mismatch ({ctx.baseline_seed} vs {ctx.post_seed})")
    if (
        ctx.baseline_config_hash
        and ctx.post_config_hash
        and ctx.baseline_config_hash != ctx.post_config_hash
    ):
        reasons.append("config_hash changed (eval-set/protocol change)")
    if reasons:
        return RuleResult(
            FailureCategory.invalid_comparison,
            "deterministic",
            "; ".join(reasons),
            "invalid_comparison",
        )
    return None


def rule_likely_instability(ctx: ClassificationContext, thr: Thresholds) -> RuleResult | None:
    cv = field_value(ctx, "gradient_norm_cv")
    if cv is not None and cv >= thr.gradient_norm_cv_instability and _regressed(ctx) is True:
        return RuleResult(
            FailureCategory.likely_instability,
            "strong_heuristic",
            f"gradient_norm_cv {cv:.2f} >= {thr.gradient_norm_cv_instability} "
            f"with a direction-aware metric regression",
            "likely_instability",
            ("gradient_norm_cv", "gradient_norm_mean", "gradient_norm_std",
             "baseline_metric", "post_change_metric"),
        )
    return None


def rule_likely_undertraining(ctx: ClassificationContext, thr: Thresholds) -> RuleResult | None:
    start = field_value(ctx, "train_loss_start")
    end = field_value(ctx, "train_loss_end")
    if start is not None and end is not None:
        slope = end - start  # proxy: total train-loss change (no per-step history emitted)
        if slope <= thr.undertraining_loss_slope and _not_improved(ctx) is True:
            return RuleResult(
                FailureCategory.likely_undertraining,
                "weak_heuristic",
                f"train loss still falling (delta={slope:.4f} <= {thr.undertraining_loss_slope}) "
                f"while val did not improve",
                "likely_undertraining",
                ("train_loss_start", "train_loss_end", "baseline_metric", "post_change_metric"),
            )
    return None


def rule_possible_overfitting(ctx: ClassificationContext, thr: Thresholds) -> RuleResult | None:
    gap = field_value(ctx, "train_val_gap")
    if gap is not None and gap >= thr.overfitting_train_val_gap and _regressed(ctx) is True:
        return RuleResult(
            FailureCategory.possible_overfitting,
            "weak_heuristic",
            f"train_val_gap {gap:.3f} >= {thr.overfitting_train_val_gap} with val regression",
            "possible_overfitting",
            ("train_val_gap", "baseline_metric", "post_change_metric"),
        )
    return None


_REGULARIZATION_TAGS = ("weight_decay", "dropout", "regulariz", "label_smooth")


def _regularization_increase(baseline: dict, changed: dict) -> str | None:
    for key, new_value in changed.items():
        lowered = key.lower()
        if any(tag in lowered for tag in _REGULARIZATION_TAGS) and key in baseline:
            try:
                if float(new_value) > float(baseline[key]):
                    return f"{key} {baseline[key]} -> {new_value}"
            except (TypeError, ValueError):
                continue
    return None


def rule_possible_over_regularization(
    ctx: ClassificationContext, thr: Thresholds
) -> RuleResult | None:
    detail = _regularization_increase(ctx.baseline_hyperparameters, ctx.changed_hyperparameters)
    if detail is None or _regressed(ctx) is not True:
        return None
    # Spec: BOTH train and val must worsen. Without train metrics we cannot confirm "both",
    # so this rule requires train-regression *evidence* rather than firing on the val signal
    # alone (which would over-claim over-regularization from a single-sided signal).
    if ctx.baseline_train_metric is None or ctx.post_train_metric is None:
        return None
    if improvement(ctx.baseline_train_metric, ctx.post_train_metric, ctx.metric_direction) >= 0:
        return None  # train did not worsen -> not over-regularization
    return RuleResult(
        FailureCategory.possible_over_regularization,
        "weak_heuristic",
        f"stronger regularization ({detail}); both train and val regressed",
        "possible_over_regularization",
        ("baseline_metric", "post_change_metric", "baseline_train_metric", "post_train_metric"),
    )


# Priority order, highest first: crash / validity categories precede heuristics.
FAILURE_RULES = [
    rule_divergence,
    rule_resource_pressure,
    rule_runtime_failure,
    rule_invalid_comparison,
    rule_likely_instability,
    rule_likely_undertraining,
    rule_possible_overfitting,
    rule_possible_over_regularization,
]


def rule_inconclusive(ctx: ClassificationContext, thr: Thresholds) -> RuleResult | None:
    """Fallback: a finished run with a valid comparison whose |improvement| is below
    the configured noise floor carries no verdict-supporting signal."""
    imp = _improvement(ctx)
    if ctx.finished and imp is not None and abs(imp) < thr.inconclusive_noise_floor:
        return RuleResult(
            FailureCategory.inconclusive,
            "weak_heuristic",
            f"|improvement| {abs(imp):.6f} < noise floor {thr.inconclusive_noise_floor}; no signal",
            "inconclusive_noise_floor",
            ("baseline_metric", "post_change_metric"),
        )
    return None
