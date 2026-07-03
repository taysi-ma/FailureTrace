"""Deterministic controlled effect-size estimation (Phase 7).

Annotates C3+ hypotheses with a magnitude + closed-form interval over their counterfactual
(controlled) trials. Purely additive to the causal-support ladder — never changes it.
"""

from .effect import (
    EstimationConfig,
    estimate_effect,
    estimate_effects,
    load_estimation_config,
)

__all__ = [
    "estimate_effect",
    "estimate_effects",
    "EstimationConfig",
    "load_estimation_config",
]
