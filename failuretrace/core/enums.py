"""Enumerations for FailureTrace core records.

All enums are :class:`enum.StrEnum` so their values serialize to plain strings in
JSON and SQLite while staying type-safe in Python. Requires Python >= 3.11.
"""

from __future__ import annotations

from enum import StrEnum


class TrialStatus(StrEnum):
    """Terminal, mutually exclusive outcomes assigned to a trial at ingestion.

    These are *terminal*: a persisted ``TrialRecord`` never transitions status after
    write (immutability). A re-run is a new trial with ``parent_trial_id`` set.

    - ``completed``      — run finished but no ratchet verdict is available.
    - ``promoted``       — ratchet's verdict: change kept / branch advanced.
    - ``rejected``       — ratchet's verdict on a finished run: discarded / reverted.
    - ``failed_runtime`` — run did not finish due to a non-OOM runtime error.
    - ``failed_oom``     — run did not finish due to an out-of-memory condition.
    - ``invalid``        — the comparison itself is unusable (e.g. baseline missing).
    - ``inconclusive``   — finished, but the signal does not support a verdict.
    """

    completed = "completed"
    promoted = "promoted"
    rejected = "rejected"
    failed_runtime = "failed_runtime"
    failed_oom = "failed_oom"
    invalid = "invalid"
    inconclusive = "inconclusive"


class FailureCategory(StrEnum):
    """Deterministic-classifier failure categories (see spec §2.2)."""

    divergence = "divergence"
    resource_pressure = "resource_pressure"
    runtime_failure = "runtime_failure"
    likely_instability = "likely_instability"
    likely_undertraining = "likely_undertraining"
    possible_overfitting = "possible_overfitting"
    possible_over_regularization = "possible_over_regularization"
    invalid_comparison = "invalid_comparison"
    inconclusive = "inconclusive"
    unknown = "unknown"


class CausalSupportLevel(StrEnum):
    """Strength of *causal* support for a claim. Ordered C0 < C1 < C2 < C3 < C4.

    A freshly ingested single trial may only assert C0/C1; C2+ is reachable solely
    through append-only ``PromotionRecord``s (replication / counterfactual evidence).
    """

    C0_observation = "C0_observation"
    C1_plausible_hypothesis = "C1_plausible_hypothesis"
    C2_replicated_effect = "C2_replicated_effect"
    C3_counterfactual_supported = "C3_counterfactual_supported"
    C4_robust_rule = "C4_robust_rule"

    @property
    def rank(self) -> int:
        """Ordinal 0..4 parsed from the ``C<n>_`` name prefix, for ordering."""
        return int(self.name[1])

    def at_least(self, other: "CausalSupportLevel") -> bool:
        """True if this level is >= ``other`` in causal strength."""
        return self.rank >= other.rank


class HypothesisSource(StrEnum):
    """Where a hypothesis came from. The LLM is strictly additive; the rule-based
    path alone must satisfy the whole pipeline."""

    rule_based = "rule_based"
    local_llm = "local_llm"
    rule_based_fallback = "rule_based_fallback"


class MetricDirection(StrEnum):
    """Whether the optimized metric improves by decreasing or increasing."""

    minimize = "minimize"
    maximize = "maximize"


class LinkType(StrEnum):
    """Kinds of append-only link records (see spec §1.3 'Link records')."""

    replication = "replication"
    validation = "validation"
    counterfactual = "counterfactual"
