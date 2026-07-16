#!/usr/bin/env python3
"""Generate REAL autoresearch trials and drive the FailureTrace governance loop.

Runs ``train.py`` a few times with different tunables (a baseline, a replicated failure,
and a controlled counterfactual "fix"), captures each real ``run.log``, ingests them into
FailureTrace, walks the promotion ladder, and prints the resulting causal-support levels +
any controlled effect estimate.

Modes:
  (default)        edit train.py constants, run training per config, capture run.log
  --from-logs DIR  skip training; ingest existing <label>.log files (CPU-only test / re-run)

Cost tip: run ``python prepare.py`` on a FREE CPU studio first (no GPU credit); attach the
A100 only for this script. See README.md.

The categories/levels you get reflect REAL dynamics. The default configs target the
resource_pressure path (oversized batch -> OOM x2 -> reduce batch), which is the
deterministic, plannable route to a real C3 + effect. Edit CONFIGS for other designs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from failuretrace import (
    CausalSupportLevel,
    MetricDirection,
    Repository,
    TrialRecord,
    TrialStatus,
    advance_promotions,
    estimate_effects,
    initialize_database,
    load_settings,
    new_trial_id,
    record_rejected_trial,
)
from failuretrace.planner import link_counterfactual_trial
from failuretrace.reporting import write_summary
from failuretrace.telemetry import parse_run_log

# role: "baseline" (sets the reference metric), "failure" (a rejected run -> hypothesis),
# "counterfactual" (a controlled fix linked to the replicated failure). The two failures
# share one intervention family across distinct (seed, commit) units -> C1->C2; the fix
# validates the plan -> C2->C3.
CONFIGS = [
    {"label": "baseline", "role": "baseline", "commit": "run-base", "seed": 42,
     "components": ["model"], "overrides": {}},
    {"label": "oom-a", "role": "failure", "commit": "run-oom-a", "seed": 43,
     "components": ["model"], "overrides": {"DEVICE_BATCH_SIZE": 512}},
    {"label": "oom-b", "role": "failure", "commit": "run-oom-b", "seed": 44,
     "components": ["model"], "overrides": {"DEVICE_BATCH_SIZE": 512}},
    {"label": "fix", "role": "counterfactual", "commit": "run-fix", "seed": 45,
     "components": ["model"], "overrides": {"DEVICE_BATCH_SIZE": 32}},
]


def _const_re(name: str) -> re.Pattern:
    return re.compile(rf"^({re.escape(name)}[ \t]*=[ \t]*)([^#\n]*)(.*)$", re.MULTILINE)


def _set_constants(train_py: Path, overrides: dict) -> str:
    """Apply {CONST: value} overrides to train.py; return original text (for restore)."""
    original = train_py.read_text(encoding="utf-8")
    text = original
    for name, value in overrides.items():
        literal = value if isinstance(value, str) else repr(value)
        text, n = _const_re(name).subn(rf"\g<1>{literal}  \g<3>", text)
        if n == 0:
            raise SystemExit(f"constant {name!r} not found in {train_py}")
    train_py.write_text(text, encoding="utf-8")
    return original


def _run_training(repo_dir: Path, log_path: Path, launcher: list[str]) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(launcher, cwd=repo_dir, stdout=log, stderr=subprocess.STDOUT, text=True)
    text = log_path.read_text(encoding="utf-8")
    print(f"    train.py exit={proc.returncode}, log={log_path}")
    return text


def _val_bpb(log_text: str) -> float | None:
    return parse_run_log(log_text).telemetry.val_metric


def _ingest_failure(repo, settings, cfg, log_text, baseline):
    parsed = parse_run_log(log_text)
    ec = {
        "git_commit": cfg["commit"], "seed": cfg["seed"], "status": "crash",
        "baseline_metric": baseline, "changed_components": cfg["components"],
        "hyperparameters": cfg["overrides"], "changed_hyperparameters": cfg["overrides"],
    }
    metrics = {"post_change_metric": parsed.telemetry.val_metric}
    diff = "# " + json.dumps(cfg["overrides"])
    rd = {"telemetry": parsed.telemetry.model_dump(), "exception_type": parsed.exception_type,
          "exception_message": parsed.exception_message, "finished": parsed.finished}
    return record_rejected_trial(ec, metrics, diff, rd, settings=settings, repository=repo)


def _save_counterfactual(repo, settings, cfg, log_text, baseline):
    """Persist the fix as a plain (completed) trial — no hypothesis; it validates a plan."""
    trial = TrialRecord(
        trial_id=new_trial_id(), git_commit=cfg["commit"], seed=cfg["seed"],
        status=TrialStatus.completed, metric_name=settings.metric.name,
        metric_direction=settings.metric.direction, baseline_metric=baseline,
        post_change_metric=_val_bpb(log_text), changed_files=["train.py"],
        changed_components=cfg["components"], hyperparameters=cfg["overrides"],
    )
    return repo.save_trial(trial)


def _load_log(cfg, args, repo_dir, reports_dir) -> str:
    if args.from_logs:
        return (Path(args.from_logs) / f"{cfg['label']}.log").read_text(encoding="utf-8")
    train_py = repo_dir / "train.py"
    print(f"[{cfg['label']}] overrides={cfg['overrides']} -> training (~5 min + compile)")
    original = _set_constants(train_py, cfg["overrides"])
    try:
        return _run_training(repo_dir, reports_dir / f"{cfg['label']}.log", args.launcher.split())
    finally:
        train_py.write_text(original, encoding="utf-8")  # always restore train.py


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="autoresearch", help="autoresearch clone dir (contains train.py)")
    ap.add_argument("--data-dir", default="ft_data")
    ap.add_argument("--reports-dir", default="ft_reports")
    ap.add_argument("--from-logs", default=None, help="ingest existing <label>.log files instead of training")
    ap.add_argument("--launcher", default="uv run train.py", help="command used to run training")
    ap.add_argument("--configs", default=None, help="JSON file with a custom CONFIGS list")
    args = ap.parse_args()

    settings = load_settings(overrides={
        "paths": {"data_dir": args.data_dir, "reports_dir": args.reports_dir},
        "ollama_enabled": False,
    }, env={})
    initialize_database(settings)
    repo = Repository(settings)
    repo_dir, reports_dir = Path(args.repo), Path(args.reports_dir)
    configs = json.loads(Path(args.configs).read_text()) if args.configs else CONFIGS

    baseline = None
    fix_trial = None
    for cfg in configs:
        log_text = _load_log(cfg, args, repo_dir, reports_dir)
        val = _val_bpb(log_text)
        if cfg["role"] == "baseline":
            baseline = val
            print(f"  baseline val_bpb = {baseline}")
        elif cfg["role"] == "failure":
            trial = _ingest_failure(repo, settings, cfg, log_text, baseline)
            print(f"  failure trial {trial.trial_id if trial else None}  val_bpb={val}")
        elif cfg["role"] == "counterfactual":
            fix_trial = _save_counterfactual(repo, settings, cfg, log_text, baseline)
            print(f"  counterfactual trial {fix_trial.trial_id}  val_bpb={val}")

    # C1 -> C2 for the replicated failure.
    advance_promotions(repo, settings)
    # Link the fix to the replicated (effective-C2) hypothesis that has a plan.
    if fix_trial is not None:
        for hyp in repo.list_hypotheses():
            level = repo.effective_causal_level(hyp.hypothesis_id)
            if level and level.at_least(CausalSupportLevel.C2_replicated_effect) \
                    and repo.list_plans_for_hypothesis(hyp.hypothesis_id):
                link_counterfactual_trial(repo, settings, hypothesis_id=hyp.hypothesis_id,
                                          counterfactual_trial_id=fix_trial.trial_id)
                break
    # C2 -> C3 + effect estimation.
    advance_promotions(repo, settings)
    estimate_effects(repo, settings)

    print("\n=== governance summary (real evidence) ===")
    for hyp in repo.list_hypotheses():
        level = repo.effective_causal_level(hyp.hypothesis_id)
        est = repo.latest_effect_estimate(hyp.hypothesis_id)
        line = f"  {hyp.category.value:18s} {level.value if level else '-'}"
        if est is not None:
            ci = f" CI[{est.ci_low:.4g},{est.ci_high:.4g}]" if est.ci_low is not None else " (n=1)"
            line += f"  effect={est.absolute_effect:+.4g}{ci}"
        print(line)
    print("report:", write_summary(repo, settings, with_plots=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
