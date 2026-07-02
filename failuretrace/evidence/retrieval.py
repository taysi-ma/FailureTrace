"""Deterministic, explainable structured retrieval of relevant prior failures.

No vector DB, no embeddings. A weighted score is computed over interpretable components
(category / component / hyperparameter-name overlap, hyperparameter range proximity,
effective causal support, recency, repeated support). All weights and settings live in
``defaults.yaml``. Every component that contributes appends one human-readable line to
``score_explanation``.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.enums import FailureCategory, MetricDirection
from ..core.models import FailureHypothesis
from ..core.settings import Settings
from ..store.repository import Repository

logger = logging.getLogger(__name__)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    weights: dict[str, float] = Field(default_factory=dict)
    recency_half_life_days: float = 30.0
    log_scale_parameters: list[str] = Field(default_factory=list)


def load_retrieval_config(settings: Settings) -> RetrievalConfig:
    return RetrievalConfig(**settings.section("retrieval"))


class InterventionContext(BaseModel):
    """What a future experiment is about to try (used to find relevant past failures)."""

    model_config = ConfigDict(extra="forbid")

    category: FailureCategory | None = None
    changed_components: list[str] = Field(default_factory=list)
    changed_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metric_direction: MetricDirection = MetricDirection.minimize


class RetrievedFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: FailureHypothesis
    relevance_score: float
    score_explanation: list[str]


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_log_scale(name: str, tokens: list[str]) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in tokens)


def _proximity(a: float, b: float, log_scale: bool) -> float:
    """Proximity in [0, 1]; 1.0 identical. Log-space params use decade distance."""
    if log_scale:
        if a <= 0 or b <= 0:
            return 0.0
        decades = abs(math.log10(a) - math.log10(b))
        return max(0.0, 1.0 - decades)  # >= 1 decade apart -> 0
    denom = max(abs(a), abs(b))
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - abs(a - b) / denom)


def _score(
    ic: InterventionContext,
    hyp: FailureHypothesis,
    trial,
    cfg: RetrievalConfig,
    repository: Repository,
    now: datetime,
) -> tuple[float, list[str]]:
    weights = cfg.weights
    lines: list[str] = []
    score = 0.0

    # category match
    if ic.category is not None and ic.category == hyp.category:
        contribution = weights.get("category_match", 0.0)
        score += contribution
        lines.append(f"category match ({hyp.category.value}): +{contribution:.2f}")

    # changed-component overlap (Jaccard)
    if trial is not None and ic.changed_components and trial.changed_components:
        a, b = set(ic.changed_components), set(trial.changed_components)
        jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
        if jaccard > 0:
            contribution = weights.get("component_match", 0.0) * jaccard
            score += contribution
            lines.append(f"component overlap {sorted(a & b)} (jaccard={jaccard:.2f}): +{contribution:.2f}")

    # hyperparameter-name overlap + range proximity
    if trial is not None and ic.changed_hyperparameters and trial.hyperparameters:
        a, b = set(ic.changed_hyperparameters), set(trial.hyperparameters)
        jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
        if jaccard > 0:
            contribution = weights.get("hyperparameter_overlap", 0.0) * jaccard
            score += contribution
            lines.append(f"hyperparameter overlap {sorted(a & b)} (jaccard={jaccard:.2f}): +{contribution:.2f}")
        proximities = []
        for key in a & b:
            va, vb = ic.changed_hyperparameters.get(key), trial.hyperparameters.get(key)
            if _is_numeric(va) and _is_numeric(vb):
                proximities.append(_proximity(float(va), float(vb), _is_log_scale(key, cfg.log_scale_parameters)))
        if proximities:
            avg = sum(proximities) / len(proximities)
            contribution = weights.get("range_proximity", 0.0) * avg
            if contribution > 0:
                score += contribution
                lines.append(f"hyperparameter range proximity (avg={avg:.2f}): +{contribution:.2f}")

    # effective causal support level
    level = repository.effective_causal_level(hyp.hypothesis_id)
    if level is not None and level.rank > 0:
        contribution = weights.get("causal_support", 0.0) * (level.rank / 4.0)
        if contribution > 0:
            score += contribution
            lines.append(f"causal support {level.value} (rank {level.rank}/4): +{contribution:.2f}")

    # recency (exponential decay)
    if trial is not None and getattr(trial, "timestamp", None) is not None:
        age_days = max(0.0, (now - trial.timestamp).total_seconds() / 86400.0)
        decay = 0.5 ** (age_days / cfg.recency_half_life_days) if cfg.recency_half_life_days > 0 else 1.0
        contribution = weights.get("recency", 0.0) * decay
        if contribution > 0:
            score += contribution
            lines.append(f"recency (age={age_days:.1f}d, decay={decay:.2f}): +{contribution:.2f}")

    # repeated support (number of promotions backing this hypothesis)
    n_promotions = len(repository.list_promotions_for_hypothesis(hyp.hypothesis_id))
    if n_promotions > 0:
        factor = 1 - 0.5 ** n_promotions
        contribution = weights.get("repeated_support", 0.0) * factor
        if contribution > 0:
            score += contribution
            lines.append(f"repeated support ({n_promotions} promotion(s), factor={factor:.2f}): +{contribution:.2f}")

    return score, lines


def retrieve_relevant_failures(
    intervention_context: InterventionContext,
    *,
    repository: Repository,
    settings: Settings,
    top_k: int = 5,
    now: datetime | None = None,
) -> list[RetrievedFailure]:
    """Return the top-k prior failure hypotheses most relevant to ``intervention_context``."""
    cfg = load_retrieval_config(settings)
    now = now or datetime.now(timezone.utc)

    scored: list[RetrievedFailure] = []
    for hyp in repository.list_hypotheses():
        trial = repository.get_trial(hyp.trial_id)
        score, explanation = _score(intervention_context, hyp, trial, cfg, repository, now)
        scored.append(
            RetrievedFailure(
                hypothesis=hyp,
                relevance_score=round(score, 4),
                score_explanation=explanation,
            )
        )
    scored.sort(key=lambda rf: rf.relevance_score, reverse=True)
    return scored[:top_k]
