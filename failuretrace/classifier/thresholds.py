"""Typed accessors over the ``thresholds`` and ``confidence`` config sections.

All thresholds and confidence tiers are injected from Settings — never hard-coded at
call sites (spec Cross-Cutting Invariant: no magic numbers in classifier logic).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..core.settings import Settings


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu_memory_ratio_resource_pressure: float
    gradient_norm_cv_instability: float
    undertraining_loss_slope: float
    overfitting_train_val_gap: float
    inconclusive_noise_floor: float
    replication_minimum_trials: int
    counterfactual_minimum_support: int


class ConfidenceTiers(BaseModel):
    model_config = ConfigDict(extra="ignore")

    deterministic: float
    strong_heuristic: float
    weak_heuristic: float
    default: float

    def value(self, tier: str) -> float:
        return float(getattr(self, tier))


def load_thresholds(settings: Settings) -> Thresholds:
    return Thresholds(**settings.section("thresholds"))


def load_confidence(settings: Settings) -> ConfidenceTiers:
    return ConfidenceTiers(**settings.section("confidence"))
