"""Input context for the classifier: normalized telemetry plus the comparison,
crash, and hyperparameter-delta signals the rules need. Built from a trial + its
baseline (parent) at ingestion (Phase 5); constructed directly in tests/fixtures.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.enums import MetricDirection
from ..telemetry.schema import TelemetryRecord


class ClassificationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    telemetry: TelemetryRecord = Field(default_factory=TelemetryRecord)
    metric_direction: MetricDirection = MetricDirection.minimize

    baseline_metric: float | None = None
    post_change_metric: float | None = None
    baseline_train_metric: float | None = None
    post_train_metric: float | None = None

    exception_type: str | None = None
    exception_message: str | None = None
    finished: bool = True

    baseline_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    changed_hyperparameters: dict[str, Any] = Field(default_factory=dict)

    # invalid-comparison signals
    baseline_metric_name: str | None = None
    post_metric_name: str | None = None
    baseline_seed: int | None = None
    post_seed: int | None = None
    baseline_config_hash: str | None = None
    post_config_hash: str | None = None
    requires_matched_seeds: bool = False


_TELEMETRY_FIELDS = frozenset(TelemetryRecord.model_fields)


def field_value(ctx: ClassificationContext, name: str) -> Any:
    """Resolve a field by name from telemetry first, then the context itself."""
    if name in _TELEMETRY_FIELDS:
        return getattr(ctx.telemetry, name)
    return getattr(ctx, name, None)


def field_present(ctx: ClassificationContext, name: str) -> bool:
    return field_value(ctx, name) is not None


def completeness(ctx: ClassificationContext, required: tuple[str, ...]) -> float:
    """Fraction of a rule's required evidence fields that are actually present."""
    if not required:
        return 1.0
    present = sum(1 for name in required if field_present(ctx, name))
    return present / len(required)
