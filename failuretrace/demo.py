"""End-to-end demo (Phase 6).

Runs the whole pipeline on the synthetic fixture set, **Ollama disabled throughout**:

    telemetry → classification → deterministic fallback hypothesis → persistence
      → retrieval for a new intervention context
      → counterfactual plan (returned, never executed)
      → replication-gate promotion on a multi-seed synthetic group (C1 → C2)
      → CLI summary + failure-map reports

Invoke via ``python -m failuretrace demo`` or ``python demo/run_demo.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .core.enums import CausalSupportLevel, FailureCategory, MetricDirection
from .core.ids import new_replication_group_id
from .core.settings import Settings, get_settings
from .classifier.context import ClassificationContext
from .evidence import InterventionContext, build_guidance, retrieve_relevant_failures, summarize_failures
from .integration.autoresearch_adapter import record_rejected_trial
from .planner import promote_replications
from .reporting import write_failure_map, write_summary
from .store.migrations import initialize_database
from .store.repository import Repository

logger = logging.getLogger(__name__)

# (scenario tag, results.tsv-style status, changed components, seed). The three
# instability rows share an intervention family across distinct seeds — the
# replication group promoted C1 -> C2. Scenario factories are imported lazily in
# run_demo() so ``import failuretrace`` never drags in the test tree.
_DEMO_TRIALS = [
    ("instability", "discard", ["optimizer"], 42),
    ("instability", "discard", ["optimizer"], 43),
    ("instability", "discard", ["optimizer"], 44),
    ("oom", "crash", ["model"], 42),
    ("undertraining", "discard", ["schedule"], 42),
    ("overfitting", "discard", ["regularization"], 42),
    ("divergence", "crash", ["optimizer"], 42),
    ("inconclusive", "discard", ["data"], 42),
]


class DemoResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trials_ingested: int
    category_counts: dict[str, int]
    replication_group_id: str
    promoted_hypothesis_id: str | None
    promotion_supporting_trials: int
    effective_level_after_promotion: str | None
    retrieval_hits: int
    retrieval_summary: str
    plan_id: str | None
    plan_primary_variable: str | None
    guidance_soft: int
    guidance_hard: int
    summary_path: str
    failure_map_path: str


def _record_from_context(
    settings: Settings,
    repository: Repository,
    ctx: ClassificationContext,
    *,
    status: str,
    changed_components: list[str],
    seed: int,
    git_commit: str,
):
    """Feed one synthetic ClassificationContext through the public record pipeline."""
    experiment_context = {
        "git_commit": git_commit,
        "seed": seed,
        "status": status,
        "baseline_metric": ctx.baseline_metric,
        "changed_components": changed_components,
        "hyperparameters": dict(ctx.changed_hyperparameters or {}),
        "changed_hyperparameters": dict(ctx.changed_hyperparameters or {}),
        "baseline_hyperparameters": dict(ctx.baseline_hyperparameters or {}),
    }
    metrics = {"post_change_metric": ctx.post_change_metric}
    diff = f"diff --git a/train.py b/train.py\n# demo change touching {', '.join(changed_components)}"
    runtime_diagnostics = {
        "telemetry": ctx.telemetry.model_dump(),
        "exception_type": ctx.exception_type,
        "exception_message": ctx.exception_message,
        "finished": ctx.finished,
    }
    trial = record_rejected_trial(
        experiment_context, metrics, diff, runtime_diagnostics,
        settings=settings, repository=repository,
    )
    hyps = repository.list_hypotheses_for_trial(trial.trial_id) if trial else []
    return trial, (hyps[0] if hyps else None)


def run_demo(settings: Settings | None = None, *, repository: Repository | None = None) -> DemoResult:
    """Execute the end-to-end demo. Ollama is forced off regardless of configuration."""
    settings = settings or get_settings()
    # "Ollama disabled throughout" — force the deterministic path (hash is unaffected).
    settings = settings.model_copy(update={"ollama_enabled": False, "enabled": True})

    initialize_database(settings)
    repository = repository or Repository(settings)

    from .tests.fixtures.scenarios import (
        divergence_nan,
        inconclusive_noise,
        instability,
        oom_crash,
        overfitting,
        undertraining,
    )

    factories = {
        "instability": instability,
        "oom": oom_crash,
        "undertraining": undertraining,
        "overfitting": overfitting,
        "divergence": divergence_nan,
        "inconclusive": inconclusive_noise,
    }

    instability_group: list = []
    for index, (tag, status, components, seed) in enumerate(_DEMO_TRIALS):
        trial, hyp = _record_from_context(
            settings, repository, factories[tag](),
            status=status, changed_components=components, seed=seed,
            git_commit=f"demo{index:03d}",
        )
        if tag == "instability" and trial is not None and hyp is not None:
            instability_group.append((trial, hyp))

    # --- replication gate: scan C1 hypotheses, promote replicated groups (C1 -> C2) ---
    # Uses the real driver (also emits append-only replication links). Single trials stay
    # C0/C1; only the multi-seed instability group has enough distinct units to reach C2.
    promotions = promote_replications(repository, settings)
    group_id = promotions[0].replication_group_id if promotions else new_replication_group_id()
    promoted_id: str | None = promotions[0].hypothesis_id if promotions else None
    supporting = len(promotions[0].supporting_trial_ids) if promotions else 0
    effective_level: str | None = None
    if promoted_id:
        level = repository.effective_causal_level(promoted_id)
        effective_level = level.value if level else None

    # --- retrieval for a NEW intervention context (a nearby instability idea) ---
    intervention = InterventionContext(
        category=FailureCategory.likely_instability,
        changed_components=["optimizer"],
        changed_hyperparameters={"MATRIX_LR": 0.07},
        metric_direction=MetricDirection.minimize,
    )
    retrieved = retrieve_relevant_failures(intervention, repository=repository, settings=settings)
    guidance = build_guidance(retrieved, settings=settings, repository=repository)

    # --- counterfactual plan (auto-created at ingestion; returned only, never executed) ---
    plan_id: str | None = None
    plan_variable: str | None = None
    plan_source_id = promoted_id or (instability_group[0][1].hypothesis_id if instability_group else None)
    if plan_source_id:
        plans = repository.list_plans_for_hypothesis(plan_source_id)
        if plans:
            plan_id = plans[0].plan_id
            plan_variable = plans[0].primary_intervention_variable

    # --- reports ---
    summary_path = write_summary(repository, settings)
    failure_map_path = write_failure_map(repository, settings)

    category_counts: dict[str, int] = {}
    for h in repository.list_hypotheses():
        category_counts[h.category.value] = category_counts.get(h.category.value, 0) + 1

    return DemoResult(
        trials_ingested=len(repository.list_trials()),
        category_counts=category_counts,
        replication_group_id=group_id,
        promoted_hypothesis_id=promoted_id,
        promotion_supporting_trials=supporting,
        effective_level_after_promotion=effective_level,
        retrieval_hits=len(retrieved),
        retrieval_summary=summarize_failures(retrieved),
        plan_id=plan_id,
        plan_primary_variable=plan_variable,
        guidance_soft=len(guidance.soft_penalties),
        guidance_hard=len(guidance.hard_constraints),
        summary_path=str(summary_path),
        failure_map_path=str(failure_map_path),
    )


def render_demo_report(result: DemoResult) -> str:
    lines = [
        "FailureTrace end-to-end demo (Ollama disabled)",
        "=" * 46,
        f"trials ingested          : {result.trials_ingested}",
        f"category distribution    : {result.category_counts}",
        "",
        "Replication gate (multi-seed instability group):",
        f"  replication_group_id   : {result.replication_group_id}",
        f"  promoted hypothesis    : {result.promoted_hypothesis_id}",
        f"  supporting trials      : {result.promotion_supporting_trials} (distinct seeds)",
        f"  effective causal level : {result.effective_level_after_promotion} "
        f"(single trials remain C0/C1 — only replication reached C2)",
        "",
        f"Retrieval for a new instability idea: {result.retrieval_hits} relevant prior failure(s)",
        result.retrieval_summary,
        "",
        f"Counterfactual plan      : {result.plan_id} (intervene on {result.plan_primary_variable}; not executed)",
        f"Search guidance          : {result.guidance_soft} soft penalty(ies), "
        f"{result.guidance_hard} hard constraint(s)",
        "",
        f"Reports written:\n  - {result.summary_path}\n  - {result.failure_map_path}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wrapper
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    print(render_demo_report(run_demo()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
