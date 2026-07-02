"""FailureTrace command-line interface.

    python -m failuretrace init
    python -m failuretrace ingest <trial.json>     # synthetic/demo ingestion
    python -m failuretrace record --commit ... --status ... --run-log run.log --repo .
    python -m failuretrace gate                     # promote replicated C1 hypotheses to C2
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
    from .planner import promote_replications

    settings = _build_settings(args)
    if not settings.replication_gate_enabled:
        print("replication gate is disabled (replication_gate_enabled: false)")
        return 0
    initialize_database(settings)
    repository = Repository(settings)
    promotions = promote_replications(repository, settings)
    if not promotions:
        print("replication gate: no hypothesis group met the promotion threshold")
        return 0
    print(f"replication gate: promoted {len(promotions)} hypothesis group(s) C1 -> C2")
    for p in promotions:
        print(f"  - {p.hypothesis_id} (group {p.replication_group_id}, "
              f"{len(p.supporting_trial_ids)} supporting trials)")
    return 0


def _cmd_guidance(args: argparse.Namespace) -> int:
    from .core.enums import FailureCategory, MetricDirection
    from .evidence import InterventionContext, summarize_guidance
    from .integration.optimizer_adapter import guidance_for

    settings = _build_settings(args)
    initialize_database(settings)
    repository = Repository(settings)

    category = FailureCategory(args.category) if args.category else None
    components = args.component or []
    ic = InterventionContext(
        category=category,
        changed_components=list(components),
        metric_direction=settings.metric.direction,
    )
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

    p_guidance = sub.add_parser("guidance", help="print search guidance for an intervention context")
    p_guidance.add_argument("--category", default=None, help="failure category to match (e.g. likely_instability)")
    p_guidance.add_argument("--component", action="append", default=None, help="changed component (repeatable)")
    p_guidance.add_argument("--top-k", type=int, default=5, dest="top_k")
    p_guidance.set_defaults(func=_cmd_guidance)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
