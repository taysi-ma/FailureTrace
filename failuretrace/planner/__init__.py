"""Deterministic counterfactual planning and the replication/promotion gate."""

from .counterfactual import plan_counterfactual
from .interventions import DEFAULT_KNOWN_VARIABLES, FAMILIES, InterventionFamily
from .replication import (
    CounterfactualResult,
    ReplicationEvidence,
    evaluate_c4,
    evaluate_counterfactual,
    evaluate_replication,
)

__all__ = [
    "plan_counterfactual",
    "InterventionFamily",
    "FAMILIES",
    "DEFAULT_KNOWN_VARIABLES",
    "ReplicationEvidence",
    "CounterfactualResult",
    "evaluate_replication",
    "evaluate_counterfactual",
    "evaluate_c4",
]
