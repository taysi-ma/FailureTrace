"""Orchestrates hypothesis generation.

- ``ollama_enabled`` false -> deterministic fallback directly (source=rule_based).
- ``ollama_enabled`` true  -> call Ollama, parse JSON, and MERGE its narrative onto the
  deterministic fallback, re-validating through the Pydantic model. On unavailability,
  timeout, invalid JSON, or validation failure -> log a warning and return the
  deterministic fallback with source=rule_based_fallback. The pipeline never crashes.

Deterministic authority: category, causal_support_level, the hard-constraint flag, ids,
trial_id, and settings_hash are always set deterministically — the LLM can enrich only
the narrative (observations/evidence/hypotheses/alternatives/missing_evidence),
confidences, and the suggested intervention / counterfactual. So an LLM can never smuggle
in C2+ support or an unjustified hard constraint.
"""

from __future__ import annotations

import json
import logging

from ..classifier.classifier import FailureClassification
from ..classifier.context import ClassificationContext
from ..classifier.thresholds import load_thresholds
from ..core.enums import HypothesisSource
from ..core.models import FailureHypothesis
from ..core.settings import Settings
from .fallback import build_fallback
from .ollama_client import OllamaClient, load_ollama_config
from .prompt import build_prompt

logger = logging.getLogger(__name__)

# Narrative fields the LLM may enrich (deterministic fields are intentionally excluded).
_LLM_LIST_FIELDS = (
    "observations",
    "evidence",
    "hypotheses",
    "alternative_explanations",
    "missing_evidence",
)


def _merge_llm(fallback: FailureHypothesis, data: dict) -> FailureHypothesis:
    """Overlay the LLM's narrative onto the deterministic fallback and re-validate."""
    fields = fallback.model_dump()
    for key in _LLM_LIST_FIELDS:
        value = data.get(key)
        if isinstance(value, list) and value:
            fields[key] = [str(item) for item in value]
    # The LLM's stated confidence is recorded for provenance only; the deterministic
    # rubric values in hypothesis_confidence / evidence_quality are NEVER overwritten
    # (Cross-Cutting Invariant 5). Bound it to [0, 1] and ignore anything else.
    llm_conf = data.get("hypothesis_confidence")
    if isinstance(llm_conf, (int, float)) and not isinstance(llm_conf, bool):
        fields["llm_confidence"] = min(1.0, max(0.0, float(llm_conf)))
    if isinstance(data.get("suggested_intervention"), dict):
        fields["suggested_intervention"] = data["suggested_intervention"]
    if isinstance(data.get("proposed_counterfactual_trial"), dict):
        fields["proposed_counterfactual_trial"] = data["proposed_counterfactual_trial"]
    # Deterministic authority — never taken from the LLM:
    fields["source"] = HypothesisSource.local_llm
    # (category, causal_support_level, should_apply_hard_constraint, ids, settings_hash stay.)
    return FailureHypothesis(**fields)


def _analyze_with_llm(
    classification: FailureClassification,
    ctx: ClassificationContext,
    *,
    trial_id: str,
    settings: Settings,
    client: OllamaClient | None,
    code_diff_summary: str | None,
    changed_components: list[str] | None,
) -> FailureHypothesis:
    # Prepared up front so any LLM failure returns a rule_based_fallback hypothesis.
    fallback = build_fallback(
        classification, ctx, trial_id=trial_id, settings=settings,
        source=HypothesisSource.rule_based_fallback,
    )
    try:
        client = client or OllamaClient(load_ollama_config(settings))
        prompt = build_prompt(
            classification, ctx,
            code_diff_summary=code_diff_summary, changed_components=changed_components,
        )
        data = json.loads(client.generate(prompt))
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a JSON object")
        hypothesis = _merge_llm(fallback, data)
        logger.info("LLM-enriched hypothesis for trial %s (source=local_llm)", trial_id)
        return hypothesis
    except Exception as exc:  # noqa: BLE001 - any failure degrades to the fallback
        logger.warning(
            "LLM analysis failed (%s: %s); using deterministic fallback",
            exc.__class__.__name__, exc,
        )
        return fallback


def analyze(
    classification: FailureClassification,
    ctx: ClassificationContext,
    *,
    trial_id: str,
    settings: Settings,
    client: OllamaClient | None = None,
    repository=None,
    code_diff_summary: str | None = None,
    changed_components: list[str] | None = None,
) -> FailureHypothesis:
    """Produce (and optionally persist) a FailureHypothesis for a trial."""
    if settings.ollama_enabled:
        hypothesis = _analyze_with_llm(
            classification, ctx, trial_id=trial_id, settings=settings, client=client,
            code_diff_summary=code_diff_summary, changed_components=changed_components,
        )
    else:
        hypothesis = build_fallback(
            classification, ctx, trial_id=trial_id, settings=settings,
            source=HypothesisSource.rule_based,
        )

    if repository is not None:
        thresholds = load_thresholds(settings)
        repository.save_hypothesis(
            hypothesis,
            telemetry=ctx.telemetry.model_dump(),
            resource_limit_gb=thresholds.resource_vram_limit_gb,
        )
    return hypothesis
