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
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
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
    {"label": "baseline", "role": "baseline", "seed": 42,
     "components": ["model"], "overrides": {}},
    {"label": "oom-a", "role": "failure", "seed": 43,
     "components": ["model"], "overrides": {"DEVICE_BATCH_SIZE": 512}},
    {"label": "oom-b", "role": "failure", "seed": 44,
     "components": ["model"], "overrides": {"DEVICE_BATCH_SIZE": 512}},
    {"label": "fix", "role": "counterfactual", "seed": 45,
     "components": ["model"], "overrides": {"DEVICE_BATCH_SIZE": 32}},
]

T4_CONFIGS = [
    {"label": "baseline", "role": "baseline", "seed": 42,
     "components": ["model"], "overrides": {}},
    {"label": "oom-a", "role": "failure", "seed": 43,
     "components": ["model"],
     "overrides": {"DEVICE_BATCH_SIZE": 512, "TOTAL_BATCH_SIZE": 2**18}},
    {"label": "oom-b", "role": "failure", "seed": 44,
     "components": ["model"],
     "overrides": {"DEVICE_BATCH_SIZE": 512, "TOTAL_BATCH_SIZE": 2**18}},
    {"label": "fix", "role": "counterfactual", "seed": 45,
     "components": ["model"], "overrides": {}},
]


@dataclass(frozen=True)
class RunArtifact:
    log_text: str
    returncode: int | None
    config_hash: str
    code_diff: str


def _const_re(name: str) -> re.Pattern:
    return re.compile(rf"^({re.escape(name)}[ \t]*=[ \t]*)([^#\n]*)(.*)$", re.MULTILINE)


def _set_run_config(train_py: Path, overrides: dict, seed: int) -> str:
    """Apply constants and the real RNG seed; return original text for restoration."""
    original = train_py.read_text(encoding="utf-8")
    text = original
    for name, value in overrides.items():
        literal = value if isinstance(value, str) else repr(value)
        text, n = _const_re(name).subn(rf"\g<1>{literal}  \g<3>", text)
        if n == 0:
            raise SystemExit(f"constant {name!r} not found in {train_py}")
    for pattern, replacement in (
        (r"torch\.manual_seed\(\d+\)", f"torch.manual_seed({seed})"),
        (r"torch\.cuda\.manual_seed\(\d+\)", f"torch.cuda.manual_seed({seed})"),
    ):
        text, n = re.subn(pattern, replacement, text, count=1)
        if n != 1:
            raise SystemExit(f"seed assignment matching {pattern!r} not found in {train_py}")
    train_py.write_text(text, encoding="utf-8")
    return original


def _run_training(repo_dir: Path, log_path: Path, launcher: list[str]) -> tuple[str, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(launcher, cwd=repo_dir, stdout=log, stderr=subprocess.STDOUT, text=True)
    text = log_path.read_text(encoding="utf-8")
    print(f"    train.py exit={proc.returncode}, log={log_path}")
    return text, proc.returncode


def _git(repo_dir: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {repo_dir}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _source_fingerprint(repo_dir: Path) -> str:
    """Hash the effective trainer/protocol while excluding the replication seed."""
    train_text = (repo_dir / "train.py").read_text(encoding="utf-8")
    train_text = re.sub(r"torch\.manual_seed\(\d+\)", "torch.manual_seed(<seed>)", train_text)
    train_text = re.sub(
        r"torch\.cuda\.manual_seed\(\d+\)", "torch.cuda.manual_seed(<seed>)", train_text,
    )
    prepare_text = (repo_dir / "prepare.py").read_text(encoding="utf-8")
    return hashlib.sha256(f"{train_text}\n{prepare_text}".encode()).hexdigest()


def _metadata_fingerprint(repo_dir: Path, cfg: dict) -> str:
    payload = {
        "base_source": _source_fingerprint(repo_dir),
        "overrides": cfg["overrides"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _ingest_failure(repo, settings, cfg, artifact, baseline, git_commit):
    parsed = parse_run_log(artifact.log_text)
    ec = {
        "git_commit": git_commit, "config_hash": artifact.config_hash,
        "seed": cfg["seed"], "status": "discard" if parsed.finished else "crash",
        "baseline_metric": baseline, "changed_components": cfg["components"],
        "hyperparameters": cfg["overrides"], "changed_hyperparameters": cfg["overrides"],
    }
    metrics = {"post_change_metric": parsed.telemetry.val_metric}
    rd = {"telemetry": parsed.telemetry.model_dump(), "exception_type": parsed.exception_type,
          "exception_message": parsed.exception_message, "finished": parsed.finished}
    return record_rejected_trial(
        ec, metrics, artifact.code_diff, rd, settings=settings, repository=repo,
    )


def _save_counterfactual(repo, settings, cfg, artifact, baseline, git_commit):
    """Persist the fix as a plain (completed) trial — no hypothesis; it validates a plan."""
    parsed = parse_run_log(artifact.log_text)
    if not parsed.finished or parsed.telemetry.val_metric is None:
        raise SystemExit(
            f"counterfactual {cfg['label']!r} did not finish with val_bpb; "
            "refusing to persist it as completed"
        )
    trial = TrialRecord(
        trial_id=new_trial_id(), git_commit=git_commit, config_hash=artifact.config_hash,
        seed=cfg["seed"],
        status=TrialStatus.completed, metric_name=settings.metric.name,
        metric_direction=settings.metric.direction, baseline_metric=baseline,
        post_change_metric=parsed.telemetry.val_metric, code_diff=artifact.code_diff,
        changed_files=["train.py"],
        changed_components=cfg["components"], hyperparameters=cfg["overrides"],
    )
    return repo.save_trial(trial)


def _require_baseline(cfg: dict, artifact: RunArtifact) -> float:
    """Return a usable baseline metric or stop before any evidence can be written."""
    parsed = parse_run_log(artifact.log_text)
    val = parsed.telemetry.val_metric
    if artifact.returncode not in (None, 0) or not parsed.finished or val is None:
        raise SystemExit(
            f"baseline {cfg['label']!r} failed or produced no val_bpb; "
            "aborting before any failure evidence is recorded"
        )
    return val


def _load_artifact(cfg, args, repo_dir, reports_dir) -> RunArtifact:
    if args.from_logs:
        log_text = (Path(args.from_logs) / f"{cfg['label']}.log").read_text(encoding="utf-8")
        return RunArtifact(
            log_text=log_text,
            returncode=None,
            config_hash=_metadata_fingerprint(repo_dir, cfg),
            code_diff="# imported config: " + json.dumps(cfg["overrides"], sort_keys=True),
        )
    train_py = repo_dir / "train.py"
    print(
        f"[{cfg['label']}] seed={cfg['seed']} overrides={cfg['overrides']} "
        "-> training (~5 min + compile)"
    )
    original = _set_run_config(train_py, cfg["overrides"], cfg["seed"])
    try:
        config_hash = _source_fingerprint(repo_dir)
        code_diff = _git(repo_dir, "diff", "--", "train.py", "prepare.py")
        log_text, returncode = _run_training(
            repo_dir, reports_dir / f"{cfg['label']}.log", args.launcher.split(),
        )
        return RunArtifact(log_text, returncode, config_hash, code_diff)
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
    ap.add_argument(
        "--profile", choices=("a100", "t4"), default="a100",
        help="built-in trial configuration (ignored when --configs is supplied)",
    )
    args = ap.parse_args()

    settings = load_settings(overrides={
        "paths": {"data_dir": args.data_dir, "reports_dir": args.reports_dir},
        "ollama_enabled": False,
    }, env={})
    initialize_database(settings)
    repo = Repository(settings)
    repo_dir, reports_dir = Path(args.repo), Path(args.reports_dir)
    configs = (
        json.loads(Path(args.configs).read_text())
        if args.configs else (T4_CONFIGS if args.profile == "t4" else CONFIGS)
    )
    git_commit = _git(repo_dir, "rev-parse", "HEAD")
    print(f"autoresearch HEAD: {git_commit}")

    baseline = None
    fix_trial = None
    for cfg in configs:
        artifact = _load_artifact(cfg, args, repo_dir, reports_dir)
        parsed = parse_run_log(artifact.log_text)
        val = parsed.telemetry.val_metric
        if cfg["role"] == "baseline":
            baseline = _require_baseline(cfg, artifact)
            print(f"  baseline val_bpb = {baseline}")
        elif cfg["role"] == "failure":
            trial = _ingest_failure(repo, settings, cfg, artifact, baseline, git_commit)
            print(f"  failure trial {trial.trial_id if trial else None}  val_bpb={val}")
        elif cfg["role"] == "counterfactual":
            fix_trial = _save_counterfactual(
                repo, settings, cfg, artifact, baseline, git_commit,
            )
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
