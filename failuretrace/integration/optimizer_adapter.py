"""Optimizer-facing adapter: produce ``SearchGuidance`` for a future search consumer.

No Optuna runtime dependency and no custom sampler (spec §5.3 / non-goals). This turns
FailureTrace's negative evidence into guidance an Optuna/TPE/BO/CMA-ES loop *could*
consult — e.g. by converting soft penalties into an additive penalty term or by pruning a
region a hard constraint forbids. FailureTrace only *produces* the guidance; it never
runs or steers a search.
"""

from __future__ import annotations

import logging

from ..core.settings import Settings
from ..evidence import (
    InterventionContext,
    SearchGuidance,
    build_guidance,
    retrieve_relevant_failures,
)
from ..store.repository import Repository

logger = logging.getLogger(__name__)


def guidance_for(
    intervention_context: InterventionContext,
    *,
    settings: Settings,
    repository: Repository,
    top_k: int = 5,
) -> SearchGuidance:
    """Retrieve relevant prior failures for a candidate point and build search guidance."""
    retrieved = retrieve_relevant_failures(
        intervention_context, repository=repository, settings=settings, top_k=top_k
    )
    return build_guidance(retrieved, settings=settings, repository=repository)


def soft_penalty_terms(guidance: SearchGuidance, *, weight: float = 1.0) -> dict[str, float]:
    """Flatten soft penalties to ``variable -> penalty`` an objective could add.

    A convenience for a future consumer only — the weight and interpretation are the
    consumer's choice; FailureTrace prescribes no optimizer behavior.
    """
    terms: dict[str, float] = {}
    for penalty in guidance.soft_penalties:
        variable = penalty.get("variable")
        if variable:
            terms[variable] = terms.get(variable, 0.0) + weight
    return terms
