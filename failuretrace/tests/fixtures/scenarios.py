"""Programmatic classification scenarios (factory functions, not opaque JSON blobs).

Each factory returns a :class:`ClassificationContext` tuned to trigger one category,
and accepts keyword overrides so later phases can perturb them (e.g. multi-seed
replication groups). ``SCENARIOS`` maps name -> (factory, expected_category).
"""

from __future__ import annotations

from failuretrace.classifier import ClassificationContext
from failuretrace.core.enums import FailureCategory, MetricDirection
from failuretrace.telemetry import TelemetryRecord


def stable_improvement(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=0.9,
        telemetry=TelemetryRecord(
            gradient_norm_cv=0.5, val_metric=0.9, peak_vram_gb=44.0, runtime_seconds=300.0
        ),
        finished=True,
    )
    base.update(over)
    return ClassificationContext(**base)


def divergence_nan(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=5.0,
        telemetry=TelemetryRecord(nan_detected=True),
        finished=False,
    )
    base.update(over)
    return ClassificationContext(**base)


def oom_crash(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=None,
        exception_type="torch.cuda.OutOfMemoryError",
        exception_message="CUDA out of memory. Tried to allocate 2.00 GiB",
        telemetry=TelemetryRecord(peak_vram_gb=80.0),
        finished=False,
    )
    base.update(over)
    return ClassificationContext(**base)


def runtime_error(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=None,
        exception_type="ValueError",
        exception_message="shape mismatch",
        finished=False,
    )
    base.update(over)
    return ClassificationContext(**base)


def instability(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=1.15,
        baseline_hyperparameters={"MATRIX_LR": 0.04},
        changed_hyperparameters={"MATRIX_LR": 0.08},
        telemetry=TelemetryRecord(
            gradient_norm_mean=1.0, gradient_norm_std=3.0, gradient_norm_max=9.0, val_metric=1.15
        ),
        finished=True,
    )
    base.update(over)
    return ClassificationContext(**base)


def undertraining(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=1.0,
        telemetry=TelemetryRecord(train_loss_start=4.0, train_loss_end=1.0, val_metric=1.0),
        finished=True,
    )
    base.update(over)
    return ClassificationContext(**base)


def overfitting(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=1.1,
        telemetry=TelemetryRecord(train_val_gap=0.2, val_metric=1.1),
        finished=True,
    )
    base.update(over)
    return ClassificationContext(**base)


def over_regularization(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=1.1,
        baseline_train_metric=0.5,
        post_train_metric=0.6,
        baseline_hyperparameters={"WEIGHT_DECAY": 0.1},
        changed_hyperparameters={"WEIGHT_DECAY": 0.3},
        telemetry=TelemetryRecord(val_metric=1.1, train_metric=0.6),
        finished=True,
    )
    base.update(over)
    return ClassificationContext(**base)


def inconclusive_noise(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=1.0005,  # |improvement| below the 0.001 noise floor
        telemetry=TelemetryRecord(val_metric=1.0005),
        finished=True,
    )
    base.update(over)
    return ClassificationContext(**base)


def invalid_comparison(**over) -> ClassificationContext:
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=None,  # baseline missing => comparison unusable
        post_change_metric=1.0,
        telemetry=TelemetryRecord(val_metric=1.0),
        finished=True,
    )
    base.update(over)
    return ClassificationContext(**base)


def missing_telemetry(**over) -> ClassificationContext:
    """Valid comparison but no telemetry signals — should degrade, never crash (T4)."""
    base = dict(
        metric_direction=MetricDirection.minimize,
        baseline_metric=1.0,
        post_change_metric=1.0,
        telemetry=TelemetryRecord(),
        finished=True,
    )
    base.update(over)
    return ClassificationContext(**base)


SCENARIOS = {
    "stable_improvement": (stable_improvement, FailureCategory.unknown),
    "divergence": (divergence_nan, FailureCategory.divergence),
    "oom": (oom_crash, FailureCategory.resource_pressure),
    "runtime_error": (runtime_error, FailureCategory.runtime_failure),
    "instability": (instability, FailureCategory.likely_instability),
    "undertraining": (undertraining, FailureCategory.likely_undertraining),
    "overfitting": (overfitting, FailureCategory.possible_overfitting),
    "over_regularization": (over_regularization, FailureCategory.possible_over_regularization),
    "inconclusive_noise": (inconclusive_noise, FailureCategory.inconclusive),
    "invalid_comparison": (invalid_comparison, FailureCategory.invalid_comparison),
}
