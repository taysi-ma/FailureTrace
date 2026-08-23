"""FailureTrace command-line interface.

    python -m failuretrace init
    python -m failuretrace ingest <trial.json>     # synthetic/demo ingestion
    python -m failuretrace record --commit ... --status ... --run-log run.log --repo .
    python -m failuretrace gate                     # promote replicated C1 hypotheses to C2
    python -m failuretrace brief --infer-from .     # prior failures for the run about to start
    python -m failuretrace guidance --category ... --component ...
    python -m failuretrace report summary|failures|map
    python -m failuretrace report trial <trial_id>

Global options: --config, --data-dir, --reports-dir, --no-ollama. Every DB-touching
command calls the idempotent ``initialize_database`` first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .core.settings import Settings, load_settings
from .store.migrations import initialize_database
from .store.repository import Repository

logger = logging.getLogger(__name__)


def _build_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    paths: dict[str, str] = {}
    if getattr(args, "data_dir", None):
        paths["data_dir"] = args.data_dir
    if getattr(args, "reports_dir", None):
        paths["reports_dir"] = args.reports_dir
    if paths:
        overrides["paths"] = paths
    if getattr(args, "no_ollama", False):
        overrides["ollama_enabled"] = False
    return load_settings(config_path=getattr(args, "config", None), overrides=overrides or None)


def _cmd_init(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    initialize_database(settings)
    print(f"initialized FailureTrace database at {settings.paths.data_dir}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .integration.autoresearch_adapter import record_rejected_trial

    settings = _build_settings(args)
    initialize_database(settings)
    repository = Repository(settings)

    data = json.loads(Path(args.trial_json).read_text(encoding="utf-8"))
    experiment_context = {
        key: data.get(key)
        for key in (
            "git_commit", "commit", "parent_trial_id", "seed", "status", "baseline_metric",
            "config_hash", "changed_files", "changed_components", "hyperparameters",
            "baseline_hyperparameters", "changed_hyperparameters", "baseline_metric_name",
            "post_metric_name", "baseline_seed", "post_seed", "baseline_config_hash",
            "post_config_hash", "requires_matched_seeds", "description",
        )
        if data.get(key) is not None
    }
    metrics = {
        "post_change_metric": data.get("post_change_metric", data.get("val_bpb")),
        "throughput": data.get("throughput"),
        "runtime_seconds": data.get("runtime_seconds"),
    }
    runtime_diagnostics = {
        "telemetry": data.get("telemetry") or {},
        "exception_type": data.get("exception_type"),
        "exception_message": data.get("exception_message"),
        "finished": data.get("finished", True),
    }
    trial = record_rejected_trial(
        experiment_context, metrics, data.get("code_diff"), runtime_diagnostics,
        settings=settings, repository=repository,
    )
    if trial is None:
        print("failuretrace is disabled (enabled: false); nothing ingested")
        return 0
    hyps = repository.list_hypotheses_for_trial(trial.trial_id)
    category = hyps[0].category.value if hyps else "unknown"
    print(f"ingested trial {trial.trial_id} (status={trial.status.value}, category={category})")
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    from .integration.autoresearch_adapter import record_from_run

    settings = _build_settings(args)
    initialize_database(settings)
    repository = Repository(settings)

    trial = record_from_run(
        settings=settings,
        repository=repository,
        commit=args.commit,
        status=args.status,
        run_log_path=args.run_log,
        repo_path=args.repo,
        branch=args.branch,
        description=args.description or "",
        baseline_metric=args.baseline_metric,
        baseline_commit=args.baseline_commit,
        seed=args.seed,
    )
    if trial is None:
        print("failuretrace is disabled (enabled: false); nothing recorded")
        return 0
    print(f"recorded trial {trial.trial_id} (commit={args.commit}, status={trial.status.value})")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import render_demo_report, run_demo

    settings = _build_settings(args)
    result = run_demo(settings)
    print(render_demo_report(result))
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from .planner import advance_promotions

    settings = _build_settings(args)
    if not settings.replication_gate_enabled:
        print("replication gate is disabled (replication_gate_enabled: false)")
        return 0
    initialize_database(settings)
    repository = Repository(settings)
    # Walk the full ladder: replication (C1->C2), counterfactual (C2->C3), C4 (C3->C4).
    result = advance_promotions(repository, settings)
    labels = {"replication": "C1 -> C2", "counterfactual": "C2 -> C3", "c4": "C3 -> C4"}
    promoted = 0
    for rung, label in labels.items():
        for p in result.get(rung, []):
            promoted += 1
            print(f"gate: promoted {p.hypothesis_id} {label} ({p.rationale})")
    # advance_promotions also estimates controlled effect sizes for C3+ hypotheses.
    estimates = result.get("effects", [])
    for e in estimates:
        ci = f" CI[{e.ci_low:.4g}, {e.ci_high:.4g}]" if e.ci_low is not None else ""
        print(f"gate: effect for {e.hypothesis_id}: {e.absolute_effect:+.4g}{ci} (n={e.n_counterfactuals})")
    if not promoted and not estimates:
        print("gate: no hypothesis met a promotion threshold")
    return 0


def _cmd_effects(args: argparse.Namespace) -> int:
    from .estimation import estimate_effects

    settings = _build_settings(args)
    initialize_database(settings)
    repository = Repository(settings)
    estimate_effects(repository, settings)  # refresh estimates (idempotent)

    rows = [
        e for e in (repository.latest_effect_estimate(h.hypothesis_id)
                    for h in repository.list_hypotheses())
        if e is not None
    ]
    if not rows:
        print("no controlled effect estimates yet (a hypothesis must reach C3)")
        return 0
    for e in rows:
        ci = f" CI[{e.ci_low:.4g}, {e.ci_high:.4g}]" if e.ci_low is not None else " (no interval, n=1)"
        print(f"{e.hypothesis_id}: effect {e.absolute_effect:+.4g}{ci}  "
              f"n={e.n_counterfactuals}  consistency={e.consistency:.2f}")
    return 0


def _parse_param(raw: str) -> tuple[str, Any]:
    """Parse a ``NAME=VALUE`` flag. VALUE is a Python literal when possible, else a string."""
    import ast

    name, sep, value = raw.partition("=")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(f"--param expects NAME=VALUE, got {raw!r}")
    try:
        parsed = ast.literal_eval(value.strip())
    except (ValueError, SyntaxError):
        parsed = value.strip()
    return name.strip(), parsed


def _intervention_context(args: argparse.Namespace, settings: Settings):
    """Build the InterventionContext describing the experiment about to be run.

    Two entry styles, combinable: explicit flags (``--category``/``--component``/
    ``--param``) and ``--infer-from <repo>``, which recovers the changed tunables by
    diffing the working tree's ``train.py`` against ``HEAD`` and maps them to components
    via the ``components:`` config section. Explicit flags win on conflict.
    """
    from .core.enums import FailureCategory
    from .evidence import InterventionContext
    from .integration.autoresearch_adapter import components_for, infer_changed_tunables

    components = list(args.component or [])
    params: dict[str, Any] = {}

    if getattr(args, "infer_from", None):
        inferred = infer_changed_tunables(args.infer_from)
        params.update(inferred)
        components.extend(c for c in components_for(inferred, settings) if c not in components)

    # Explicit --param overrides an inferred value for the same knob.
    for name, value in (args.param or []):
        params[name] = value

    ic = InterventionContext(
        category=FailureCategory(args.category) if args.category else None,
        changed_components=components,
        changed_hyperparameters=params,
        metric_direction=settings.metric.direction,
    )
    # Recency contributes to every stored hypothesis, so an empty context would return
    # arbitrary recent failures dressed up as relevant. Report that instead.
    has_signal = bool(ic.category or ic.changed_components or ic.changed_hyperparameters)
    return ic, has_signal


def _cmd_brief(args: argparse.Namespace) -> int:
    from .evidence import NO_CONTEXT_MESSAGE, brief_for, render_brief

    settings = _build_settings(args)
    if not settings.enabled:
        print("failuretrace is disabled (enabled: false); no brief produced")
        return 0
    initialize_database(settings)
    repository = Repository(settings)

    ic, has_signal = _intervention_context(args, settings)
    if not has_signal:
        print(
            "no experiment context given: pass --category, --component, --param NAME=VALUE, "
            "or --infer-from <repo>"
        )
        return 2

    brief = brief_for(ic, settings=settings, repository=repository, top_k=args.top_k)
    if brief.is_empty and args.format != "json":
        print(NO_CONTEXT_MESSAGE)
        return 0
    print(render_brief(brief, fmt=args.format, settings=settings))
    return 0


def _cmd_guidance(args: argparse.Namespace) -> int:
    from .evidence import summarize_guidance
    from .integration.optimizer_adapter import guidance_for

    settings = _build_settings(args)
    if not settings.enabled:
        print("failuretrace is disabled (enabled: false); no guidance produced")
        return 0
    initialize_database(settings)
    repository = Repository(settings)

    ic, _ = _intervention_context(args, settings)
    guidance = guidance_for(ic, settings=settings, repository=repository, top_k=args.top_k)
    print(f"search guidance: {summarize_guidance(guidance)}")
    for hc in guidance.hard_constraints:
        print(f"  [HARD] {hc.get('category', '?')}: {hc.get('reason', '')}")
    for sp in guidance.soft_penalties:
        print(f"  [soft] {sp.get('category', '?')}: {sp.get('reason', '')}")
    for w in guidance.warnings:
        print(f"  [warn] {w}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .reporting import write_failure_map, write_summary, write_trial_report

    settings = _build_settings(args)
    initialize_database(settings)
    repository = Repository(settings)

    kind = args.kind
    if kind == "summary":
        path = write_summary(repository, settings)
    elif kind in ("failures", "map"):
        path = write_failure_map(repository, settings)
    elif kind == "trial":
        if not args.trial_id:
            print("report trial requires a <trial_id>", file=sys.stderr)
            return 2
        path = write_trial_report(repository, settings, args.trial_id)
    else:  # pragma: no cover - argparse choices guard this
        print(f"unknown report kind: {kind}", file=sys.stderr)
        return 2

    print(f"wrote {path}")
    print(path.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="failuretrace", description="Failure-aware experiment governance.")
    parser.add_argument("--config", help="path to a full config YAML (defaults to packaged defaults.yaml)")
    parser.add_argument("--data-dir", help="override paths.data_dir")
    parser.add_argument("--reports-dir", help="override paths.reports_dir")
    parser.add_argument("--no-ollama", action="store_true", help="force ollama_enabled off (deterministic/offline)")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create/upgrade the database (idempotent)").set_defaults(func=_cmd_init)

    sub.add_parser("demo", help="run the end-to-end synthetic demo (Ollama disabled)").set_defaults(func=_cmd_demo)

    p_ingest = sub.add_parser("ingest", help="ingest a synthetic/demo trial JSON")
    p_ingest.add_argument("trial_json", help="path to a trial JSON file")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_record = sub.add_parser("record", help="record a rejected/crashed autoresearch run (live hook)")
    p_record.add_argument("--commit", required=True)
    p_record.add_argument("--status", required=True, choices=["discard", "crash", "keep"])
    p_record.add_argument("--run-log", required=True, help="path to run.log")
    p_record.add_argument("--repo", default=".", help="autoresearch repo path (for git diff/hyperparams)")
    p_record.add_argument("--branch", default=None)
    p_record.add_argument("--description", default="")
    p_record.add_argument("--baseline-metric", type=float, default=None, dest="baseline_metric")
    p_record.add_argument("--baseline-commit", default=None, dest="baseline_commit")
    p_record.add_argument("--seed", type=int, default=42)
    p_record.set_defaults(func=_cmd_record)

    p_report = sub.add_parser("report", help="write a report artifact and print it")
    p_report.add_argument("kind", choices=["summary", "failures", "map", "trial"])
    p_report.add_argument("trial_id", nargs="?", default=None)
    p_report.set_defaults(func=_cmd_report)

    sub.add_parser(
        "gate", help="run the replication gate: promote replicated C1 hypotheses to C2"
    ).set_defaults(func=_cmd_gate)

    sub.add_parser(
        "effects", help="estimate and list controlled effect sizes for C3+ hypotheses"
    ).set_defaults(func=_cmd_effects)

    def _add_context_args(p: argparse.ArgumentParser) -> None:
        """Flags describing the experiment about to be run (shared by brief + guidance)."""
        p.add_argument("--category", default=None, help="failure category to match (e.g. likely_instability)")
        p.add_argument("--component", action="append", default=None, help="changed component (repeatable)")
        p.add_argument(
            "--param", action="append", type=_parse_param, default=None, metavar="NAME=VALUE",
            help="hyperparameter this experiment sets, e.g. --param MATRIX_LR=0.08 (repeatable)",
        )
        p.add_argument(
            "--infer-from", default=None, dest="infer_from", metavar="REPO",
            help="infer changed tunables by diffing REPO/train.py against git HEAD",
        )
        p.add_argument("--top-k", type=int, default=5, dest="top_k")

    p_guidance = sub.add_parser("guidance", help="print search guidance for an intervention context")
    _add_context_args(p_guidance)
    p_guidance.set_defaults(func=_cmd_guidance)

    p_brief = sub.add_parser(
        "brief", help="print bounded prior-failure evidence for the experiment about to run"
    )
    _add_context_args(p_brief)
    p_brief.add_argument("--format", default="markdown", choices=["markdown", "text", "json"])
    p_brief.set_defaults(func=_cmd_brief)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
