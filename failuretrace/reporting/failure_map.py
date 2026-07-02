"""Failure map: a grouped table (category x aggregates) plus an optional scatter.

Simple by design (spec §5.2): a per-category rollup of counts, mean confidence, the
effective causal-support mix, and example trials — rendered as a markdown table and,
when matplotlib is present, a metric-delta vs peak-VRAM scatter colored by category.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..core.enums import CausalSupportLevel
from ..core.settings import Settings, improvement
from ..store.repository import Repository

logger = logging.getLogger(__name__)


class FailureMapRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    count: int
    mean_confidence: float | None = None
    effective_levels: dict[str, int] = Field(default_factory=dict)
    example_trials: list[str] = Field(default_factory=list)


def build_failure_map(repository: Repository, settings: Settings) -> list[FailureMapRow]:
    hyps = repository.list_hypotheses()
    by_category: dict[str, list] = defaultdict(list)
    for h in hyps:
        by_category[h.category.value].append(h)

    rows: list[FailureMapRow] = []
    for category, group in by_category.items():
        confidences = [h.hypothesis_confidence for h in group]
        levels = Counter(
            (repository.effective_causal_level(h.hypothesis_id) or h.causal_support_level).value
            for h in group
        )
        rows.append(
            FailureMapRow(
                category=category,
                count=len(group),
                mean_confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
                effective_levels=dict(levels),
                example_trials=[h.trial_id for h in group[:3]],
            )
        )
    rows.sort(key=lambda r: r.count, reverse=True)
    return rows


def render_failure_map_text(rows: list[FailureMapRow]) -> str:
    lines = ["# FailureTrace — failure map", ""]
    if not rows:
        lines.append("_No failure hypotheses recorded yet._")
        return "\n".join(lines) + "\n"
    lines.append("| category | count | mean confidence | effective levels | example trials |")
    lines.append("|---|---:|---:|---|---|")
    for r in rows:
        levels = ", ".join(f"{k}:{v}" for k, v in sorted(r.effective_levels.items())) or "—"
        examples = ", ".join(r.example_trials) or "—"
        lines.append(f"| {r.category} | {r.count} | {r.mean_confidence} | {levels} | {examples} |")
    lines.append("")
    lines.append("> Counts and confidences are descriptive. A category appearing here is a "
                 "**plausible** failure pattern, not a validated cause, unless its effective "
                 "level is C2+.")
    return "\n".join(lines) + "\n"


def write_failure_map(repository: Repository, settings: Settings, *, with_plots: bool = True) -> Path:
    rows = build_failure_map(repository, settings)
    reports_dir = Path(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "failure_map.md"
    path.write_text(render_failure_map_text(rows), encoding="utf-8")

    if with_plots:
        from .plots import scatter_plot

        direction = settings.metric.direction
        points: list[tuple[float, float, str]] = []
        trials = {t.trial_id: t for t in repository.list_trials()}
        for h in repository.list_hypotheses():
            trial = trials.get(h.trial_id)
            if trial is None or trial.baseline_metric is None or trial.post_change_metric is None:
                continue
            if trial.peak_vram_gb is None:
                continue
            delta = improvement(trial.baseline_metric, trial.post_change_metric, direction)
            points.append((delta, trial.peak_vram_gb, h.category.value))
        scatter_plot(
            points, "Failure map: improvement vs peak VRAM", reports_dir / "failure_map.png",
            xlabel="direction-aware improvement", ylabel="peak VRAM (GB)",
        )

    logger.info("wrote failure map to %s", path)
    return path
