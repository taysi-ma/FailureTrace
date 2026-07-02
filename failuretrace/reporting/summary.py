"""Overall governance summary.

Aggregates trials + hypotheses into counts and, crucially, groups findings by
**effective** causal support level (post-promotion) under headers that never present
C0/C1 as causal conclusions.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..classifier.thresholds import load_thresholds
from ..core.enums import CausalSupportLevel, FailureCategory, TrialStatus
from ..core.settings import Settings
from ..store.repository import Repository

logger = logging.getLogger(__name__)

# Categories for which a soft search penalty is meaningful (actionable knobs exist).
ACTIONABLE = frozenset(
    {
        FailureCategory.likely_instability,
        FailureCategory.likely_undertraining,
        FailureCategory.possible_overfitting,
        FailureCategory.possible_over_regularization,
        FailureCategory.resource_pressure,
    }
)

_REJECTED_STATUSES = frozenset(
    {
        TrialStatus.rejected,
        TrialStatus.failed_oom,
        TrialStatus.failed_runtime,
        TrialStatus.invalid,
    }
)

# (level, section header). Headers are deliberately explicit about non-causality.
_LEVEL_SECTIONS = [
    (CausalSupportLevel.C0_observation, "Observations (C0 — raw signal, NOT causal)"),
    (CausalSupportLevel.C1_plausible_hypothesis, "Plausible hypotheses (C1 — NOT causally validated)"),
    (CausalSupportLevel.C2_replicated_effect, "Replicated effects (C2 — replicated; not yet counterfactually validated)"),
    (CausalSupportLevel.C3_counterfactual_supported, "Counterfactual-supported effects (C3)"),
    (CausalSupportLevel.C4_robust_rule, "Robust rules (C4 — rare)"),
]


class ReportSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_trials: int
    total_hypotheses: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    category_counts: dict[str, int] = Field(default_factory=dict)
    effective_level_counts: dict[str, int] = Field(default_factory=dict)
    repeated_patterns: dict[str, int] = Field(default_factory=dict)
    rejection_causes: dict[str, int] = Field(default_factory=dict)
    hard_constraints: list[dict] = Field(default_factory=list)
    soft_penalties: list[dict] = Field(default_factory=list)
    mean_hypothesis_confidence: float | None = None
    mean_evidence_quality: float | None = None
    findings_by_level: dict[str, list[str]] = Field(default_factory=dict)


def build_summary(repository: Repository, settings: Settings) -> ReportSummary:
    thresholds = load_thresholds(settings)
    trials = repository.list_trials()
    hyps = repository.list_hypotheses()
    status_of = {t.trial_id: t.status for t in trials}

    status_counts = dict(Counter(t.status.value for t in trials))
    category_counts = dict(Counter(h.category.value for h in hyps))

    effective = {
        h.hypothesis_id: (repository.effective_causal_level(h.hypothesis_id) or h.causal_support_level)
        for h in hyps
    }
    effective_level_counts = dict(Counter(level.value for level in effective.values()))

    repeated = {c: n for c, n in category_counts.items() if n >= thresholds.replication_minimum_trials}
    rejection_causes = dict(
        Counter(h.category.value for h in hyps if status_of.get(h.trial_id) in _REJECTED_STATUSES)
    )

    hard: list[dict] = []
    soft: list[dict] = []
    for h in hyps:
        level = effective[h.hypothesis_id]
        if h.should_apply_hard_constraint or level.at_least(CausalSupportLevel.C2_replicated_effect):
            hard.append({
                "hypothesis_id": h.hypothesis_id,
                "category": h.category.value,
                "reason": "C2+ effective evidence" if level.at_least(CausalSupportLevel.C2_replicated_effect)
                else "deterministic-repeated / objective resource limit",
            })
        elif h.category in ACTIONABLE:
            soft.append({"hypothesis_id": h.hypothesis_id, "category": h.category.value})

    confidences = [h.hypothesis_confidence for h in hyps]
    qualities = [h.evidence_quality for h in hyps]

    findings_by_level: dict[str, list[str]] = defaultdict(list)
    for h in hyps:
        level = effective[h.hypothesis_id]
        headline = h.hypotheses[0] if h.hypotheses else "—"
        findings_by_level[level.value].append(
            f"{h.category.value}: {headline} (trial {h.trial_id}, confidence={h.hypothesis_confidence:.2f})"
        )

    return ReportSummary(
        total_trials=len(trials),
        total_hypotheses=len(hyps),
        status_counts=status_counts,
        category_counts=category_counts,
        effective_level_counts=effective_level_counts,
        repeated_patterns=repeated,
        rejection_causes=rejection_causes,
        hard_constraints=hard,
        soft_penalties=soft,
        mean_hypothesis_confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
        mean_evidence_quality=round(sum(qualities) / len(qualities), 4) if qualities else None,
        findings_by_level=dict(findings_by_level),
    )


def _counts_block(counts: dict[str, int]) -> str:
    if not counts:
        return "_(none)_\n"
    return "".join(f"- **{k}**: {v}\n" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


def render_summary_text(summary: ReportSummary) -> str:
    lines: list[str] = ["# FailureTrace — governance summary", ""]
    lines.append(f"Trials recorded: **{summary.total_trials}**  ·  "
                 f"failure hypotheses: **{summary.total_hypotheses}**")
    lines.append("")

    lines.append("## Trial counts by status")
    lines.append(_counts_block(summary.status_counts))
    lines.append("## Failure category distribution")
    lines.append(_counts_block(summary.category_counts))
    lines.append("## Repeated failure patterns (>= replication minimum)")
    lines.append(_counts_block(summary.repeated_patterns))
    lines.append("## Rejection causes")
    lines.append(_counts_block(summary.rejection_causes))
    lines.append("## Effective causal support distribution (post-promotion)")
    lines.append(_counts_block(summary.effective_level_counts))

    lines.append("## Search guidance recommendations")
    lines.append("### Soft penalties (default)")
    if summary.soft_penalties:
        lines += [f"- {p['category']} (hypothesis {p['hypothesis_id']})" for p in summary.soft_penalties]
        lines.append("")
    else:
        lines.append("_(none)_\n")
    lines.append("### Hard constraints (deterministic-repeated or C2+ ONLY)")
    if summary.hard_constraints:
        lines += [f"- {c['category']} — {c['reason']} (hypothesis {c['hypothesis_id']})"
                  for c in summary.hard_constraints]
        lines.append("")
    else:
        lines.append("_(none)_\n")

    lines.append("## Confidence summary")
    lines.append(f"- mean hypothesis confidence: {summary.mean_hypothesis_confidence}")
    lines.append(f"- mean evidence quality: {summary.mean_evidence_quality}")
    lines.append("")

    lines.append("## Findings by causal support level")
    lines.append("> C0/C1 are observations and plausible hypotheses only — **not** causal "
                 "conclusions. Causal claims require C2+ (replication) and C3+ (counterfactual).")
    lines.append("")
    for level, header in _LEVEL_SECTIONS:
        lines.append(f"### {header}")
        items = summary.findings_by_level.get(level.value, [])
        if items:
            lines += [f"- {item}" for item in items]
        else:
            lines.append("_(none)_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_summary(repository: Repository, settings: Settings, *, with_plots: bool = True) -> Path:
    summary = build_summary(repository, settings)
    reports_dir = Path(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "summary.md"
    path.write_text(render_summary_text(summary), encoding="utf-8")

    if with_plots:
        from .plots import bar_plot

        bar_plot(summary.status_counts, "Trials by status", reports_dir / "trials_by_status.png")
        bar_plot(summary.category_counts, "Failure categories", reports_dir / "failure_categories.png")
        bar_plot(summary.effective_level_counts, "Effective causal support", reports_dir / "causal_levels.png")

    logger.info("wrote summary report to %s", path)
    return path
