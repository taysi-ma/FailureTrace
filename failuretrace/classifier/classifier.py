"""The deterministic classifier orchestrator.

Confidence rubric (deterministic — no arbitrary floats): each rule declares a tier
(``deterministic`` -> 0.95, ``strong_heuristic`` -> 0.7, ``weak_heuristic`` -> 0.5,
``default`` -> 0.3, all from ``defaults.yaml``); the value is then capped by evidence
completeness, i.e. multiplied by ``(present required fields / required fields)`` for
that rule. Failure rules are evaluated in priority order; the highest-priority rule that
fires wins, and the categories of any other fired rules become
``alternative_categories``. If no failure rule fires, the result degrades to
``inconclusive`` (finished + |improvement| below the noise floor) or ``unknown``.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from ..core.enums import FailureCategory
from ..core.settings import Settings
from .context import ClassificationContext, completeness
from .rules import FAILURE_RULES, RuleResult, rule_inconclusive
from .thresholds import ConfidenceTiers, load_confidence, load_thresholds

logger = logging.getLogger(__name__)


class FailureClassification(BaseModel):
    """Explainable output of the deterministic classifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: FailureCategory
    confidence: float = Field(ge=0.0, le=1.0)
    observations: list[str]
    triggered_rules: list[str]
    alternative_categories: list[FailureCategory] = Field(default_factory=list)
    settings_hash: str


def _confidence(result: RuleResult, ctx: ClassificationContext, tiers: ConfidenceTiers) -> float:
    value = tiers.value(result.tier) * completeness(ctx, result.required_fields)
    return round(min(1.0, max(0.0, value)), 4)


def classify(ctx: ClassificationContext, settings: Settings) -> FailureClassification:
    thresholds = load_thresholds(settings)
    tiers = load_confidence(settings)
    settings_hash = settings.settings_hash()

    fired = [r for r in (rule(ctx, thresholds) for rule in FAILURE_RULES) if r is not None]

    if fired:
        winner = fired[0]  # FAILURE_RULES is priority-ordered; first fired = winner
        alternatives: list[FailureCategory] = []
        for result in fired[1:]:
            if result.category != winner.category and result.category not in alternatives:
                alternatives.append(result.category)
        classification = FailureClassification(
            category=winner.category,
            confidence=_confidence(winner, ctx, tiers),
            observations=[r.observation for r in fired],
            triggered_rules=[r.triggered_rule for r in fired],
            alternative_categories=alternatives,
            settings_hash=settings_hash,
        )
        logger.info(
            "classified as %s (confidence=%.2f, rules=%s)",
            classification.category,
            classification.confidence,
            classification.triggered_rules,
        )
        return classification

    inconclusive = rule_inconclusive(ctx, thresholds)
    if inconclusive is not None:
        return FailureClassification(
            category=inconclusive.category,
            confidence=_confidence(inconclusive, ctx, tiers),
            observations=[inconclusive.observation],
            triggered_rules=[inconclusive.triggered_rule],
            settings_hash=settings_hash,
        )

    return FailureClassification(
        category=FailureCategory.unknown,
        confidence=round(tiers.default, 4),
        observations=["no rule matched the available evidence"],
        triggered_rules=["unknown_fallback"],
        settings_hash=settings_hash,
    )
