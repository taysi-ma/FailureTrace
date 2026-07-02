"""Per-trial report: the trial, its hypotheses (with effective level), and any plans."""

from __future__ import annotations

import logging
from pathlib import Path

from ..core.settings import Settings, improvement
from ..store.repository import Repository

logger = logging.getLogger(__name__)


def render_trial_text(repository: Repository, settings: Settings, trial_id: str) -> str:
    trial = repository.get_trial(trial_id)
    if trial is None:
        return f"# Trial {trial_id}\n\n_Not found._\n"

    lines = [f"# Trial {trial.trial_id}", ""]
    lines.append(f"- status: **{trial.status.value}**")
    lines.append(f"- git commit: {trial.git_commit}  ·  seed: {trial.seed}")
    lines.append(f"- metric: {trial.metric_name} ({trial.metric_direction.value})")
    if trial.baseline_metric is not None and trial.post_change_metric is not None:
        delta = improvement(trial.baseline_metric, trial.post_change_metric, trial.metric_direction)
        verdict = "improvement" if delta > 0 else ("regression" if delta < 0 else "no change")
        lines.append(f"- baseline → post: {trial.baseline_metric} → {trial.post_change_metric} "
                     f"(direction-aware improvement = {delta:+.6f}, {verdict})")
    if trial.exception_type:
        lines.append(f"- exception: {trial.exception_type}: {trial.exception_message}")
    if trial.peak_vram_gb is not None:
        lines.append(f"- peak VRAM: {trial.peak_vram_gb} GB")
    if trial.changed_components:
        lines.append(f"- changed components: {', '.join(trial.changed_components)}")
    lines.append("")

    hyps = repository.list_hypotheses_for_trial(trial_id)
    lines.append("## Failure hypotheses")
    if not hyps:
        lines.append("_(none)_")
    for h in hyps:
        effective = repository.effective_causal_level(h.hypothesis_id) or h.causal_support_level
        lines.append(f"### {h.category.value}  ·  effective level: {effective.value}")
        lines.append(f"- source: {h.source.value}  ·  confidence: {h.hypothesis_confidence:.2f}  "
                     f"·  evidence quality: {h.evidence_quality:.2f}")
        if h.observations:
            lines.append(f"- observations: {'; '.join(h.observations)}")
        if h.hypotheses:
            lines.append(f"- hypotheses: {'; '.join(h.hypotheses)}")
        if h.alternative_explanations:
            lines.append(f"- alternative explanations: {'; '.join(h.alternative_explanations)}")
        if h.missing_evidence:
            lines.append(f"- missing evidence: {'; '.join(h.missing_evidence)}")
        plans = repository.list_plans_for_hypothesis(h.hypothesis_id)
        for plan in plans:
            coupled = f" + {plan.optional_coupled_stabilization_variable}" if plan.optional_coupled_stabilization_variable else ""
            lines.append(f"- counterfactual plan: intervene on **{plan.primary_intervention_variable}"
                         f"{coupled}**; expected-if-true: {plan.expected_outcome_if_hypothesis_true}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_trial_report(repository: Repository, settings: Settings, trial_id: str) -> Path:
    reports_dir = Path(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"trial_{trial_id}.md"
    path.write_text(render_trial_text(repository, settings, trial_id), encoding="utf-8")
    logger.info("wrote trial report to %s", path)
    return path
