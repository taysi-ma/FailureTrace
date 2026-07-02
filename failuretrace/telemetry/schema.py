"""Normalized telemetry schema. Every field is optional — partial metrics must be
accepted gracefully (a run may emit only a subset). GPU metrics are optional and the
schema never requires CUDA; everything is CPU-safe.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class TelemetryRecord(BaseModel):
    """A normalized, trainer-agnostic view of one trial's telemetry."""

    model_config = ConfigDict(extra="forbid")

    # loss / metric signals
    train_loss_start: float | None = None
    train_loss_end: float | None = None
    val_metric: float | None = None
    train_metric: float | None = None

    # gradient statistics
    gradient_norm_mean: float | None = None
    gradient_norm_std: float | None = None
    gradient_norm_max: float | None = None
    gradient_norm_cv: float | None = None  # coefficient of variation (std / mean)

    # stability flags
    loss_spike_count: int | None = None
    nan_detected: bool | None = None
    inf_detected: bool | None = None

    # resource / throughput
    peak_vram_gb: float | None = None
    gpu_memory_ratio: float | None = None
    throughput: float | None = None
    runtime_seconds: float | None = None

    # schedules / summaries
    learning_rate_history: list[float] | None = None
    train_val_gap: float | None = None
    parameter_norm_summary: dict[str, float] | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cv = data.get("gradient_norm_cv")
            mean = data.get("gradient_norm_mean")
            std = data.get("gradient_norm_std")
            if cv is None and mean not in (None, 0) and std is not None:
                data["gradient_norm_cv"] = std / mean
        return data

    def present_fields(self) -> set[str]:
        """Names of fields that carry a value (non-None)."""
        return {name for name in type(self).model_fields if getattr(self, name) is not None}
