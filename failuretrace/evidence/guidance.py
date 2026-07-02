"""Turn retrieved failures into search guidance.

Soft penalties are the default. Hard constraints are emitted ONLY for repeated
deterministic resource failures or C2+ (effective) evidence. Inconclusive nearby results
produce context (a warning) but never a constraint.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from ..classifier.thresholds import load_thresholds
from ..core.enums import CausalSupportLevel, FailureCategory
from ..core.settings import Settings
from ..store.repository import Repository
from .retrieval import RetrievedFailure

logger = logging.getLogger(__name__)


class SearchGuidance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    soft_penalties: list[dict] = Field(default_factory=list)
    hard_constraints: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    relevant_failure_hypotheses: list[str] = Field(default_factory=list)


def build_guidance(
    retrieved: list[RetrievedFailure],
    *,
    settings: Settings,
    repository: Repository,
) -> SearchGuidance:
    thresholds = load_thresholds(settings)
    min_repeat = thresholds.replication_minimum_trials

    soft: list[dict] = []
    hard: list[dict] = []
    warnings: list[str] = []
    ids: list[str] = []

    def _distinct_trials(group: list[RetrievedFailure]) -> int:
        """Count *distinct source trials* behind a group of hypotheses. The same physical
        failure recorded twice (e.g. live hook + offline backfill of one commit) collapses
        to a single observation, so a duplicate can never manufacture "repeated" evidence."""
        identities: set[tuple[str, str]] = set()
        for rf in group:
            trial = repository.get_trial(rf.hypothesis.trial_id)
            if trial is not None and trial.git_commit:
                identities.add(("commit", trial.git_commit))
            else:
                identities.add(("trial", rf.hypothesis.trial_id))
        return len(identities)

    by_category: dict[FailureCategory, list[RetrievedFailure]] = defaultdict(list)
    for rf in retrieved:
        by_category[rf.hypothesis.category].append(rf)
        ids.append(rf.hypothesis.hypothesis_id)

    # Repeated instability -> warning + soft penalty (never hard on its own).
    instability = by_category.get(FailureCategory.likely_instability, [])
    n_instability = _distinct_trials(instability)
    if n_instability >= min_repeat:
        warnings.append(f"repeated instability across {n_instability} similar prior trials")
        soft.append({
            "kind": "soft_penalty", "category": "likely_instability",
            "variable": "optimizer.lr", "reason": f"repeated instability in {n_instability} nearby trials",
        })
    elif instability:
        soft.append({
            "kind": "soft_penalty", "category": "likely_instability",
            "variable": "optimizer.lr", "reason": "prior instability nearby",
        })

    # Repeated deterministic OOM -> hard resource constraint; single -> soft.
    resource = by_category.get(FailureCategory.resource_pressure, [])
    n_resource = _distinct_trials(resource)
    if n_resource >= min_repeat:
        hard.append({
            "kind": "hard_constraint", "category": "resource_pressure",
            "variable": "DEVICE_BATCH_SIZE",
            "reason": f"repeated deterministic OOM in {n_resource} nearby trials",
        })
    elif resource:
        soft.append({
            "kind": "soft_penalty", "category": "resource_pressure",
            "variable": "DEVICE_BATCH_SIZE", "reason": "prior memory pressure nearby",
        })

    # C2+ (effective) evidence -> hard constraint.
    for rf in retrieved:
        level = repository.effective_causal_level(rf.hypothesis.hypothesis_id)
        if level is not None and level.at_least(CausalSupportLevel.C2_replicated_effect):
            hard.append({
                "kind": "hard_constraint", "hypothesis_id": rf.hypothesis.hypothesis_id,
                "category": rf.hypothesis.category.value, "reason": f"C2+ evidence ({level.value})",
            })

    # Inconclusive -> context only.
    inconclusive = by_category.get(FailureCategory.inconclusive, [])
    if inconclusive:
        warnings.append(
            f"{len(inconclusive)} inconclusive nearby result(s) — context only, no constraint"
        )

    return SearchGuidance(
        soft_penalties=soft, hard_constraints=hard, warnings=warnings,
        relevant_failure_hypotheses=ids,
    )
