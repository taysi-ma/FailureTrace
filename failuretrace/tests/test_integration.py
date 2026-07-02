"""Integration adapter tests: public API, live hook, offline backfill, optimizer feed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from failuretrace import (
    InterventionContext,
    record_from_run,
    record_rejected_trial,
    render_program_md_hook,
)
from failuretrace.core.enums import FailureCategory, TrialStatus
from failuretrace.integration.autoresearch_adapter import ingest_results_tsv
from failuretrace.integration.optimizer_adapter import guidance_for, soft_penalty_terms

_RUN_LOG = """\
compiling...
step 953 | loss: 2.10 | lrm: 0.02 | dt: 0.31 | tok/sec: 1.6e6 | mfu: 39.8 | epoch: 1
---
val_bpb:          1.050000
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
"""

_TRAIN_PY_BASE = "MATRIX_LR = 0.04\nWEIGHT_DECAY = 0.1\nDEPTH = 8\nprint('train')\n"
_TRAIN_PY_EXP = "MATRIX_LR = 0.08\nWEIGHT_DECAY = 0.1\nDEPTH = 8\nprint('train')\n"


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _make_git_repo(root: Path) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    (root / "train.py").write_text(_TRAIN_PY_BASE, encoding="utf-8")
    _git(root, "add", "train.py")
    _git(root, "commit", "-q", "-m", "baseline")
    base = _git(root, "rev-parse", "HEAD")
    (root / "train.py").write_text(_TRAIN_PY_EXP, encoding="utf-8")
    _git(root, "commit", "-q", "-am", "raise LR")
    exp = _git(root, "rev-parse", "HEAD")
    return base, exp


# --- public API -----------------------------------------------------------------
def test_record_rejected_trial_persists_and_classifies(make_env):
    settings, repo = make_env(ollama_enabled=False)
    trial = record_rejected_trial(
        {"git_commit": "c0ffee1", "seed": 42, "status": "discard", "baseline_metric": 1.0,
         "changed_components": ["optimizer"], "hyperparameters": {"MATRIX_LR": 0.08},
         "changed_hyperparameters": {"MATRIX_LR": 0.08}, "baseline_hyperparameters": {"MATRIX_LR": 0.04}},
        {"post_change_metric": 1.15},
        "diff --git a/train.py b/train.py\n+MATRIX_LR = 0.08",
        {"telemetry": {"gradient_norm_mean": 1.0, "gradient_norm_std": 3.0, "val_metric": 1.15}, "finished": True},
        settings=settings, repository=repo,
    )
    assert trial is not None
    assert trial.status == TrialStatus.rejected
    assert repo.get_trial(trial.trial_id) is not None
    assert repo.json.exists(trial.trial_id)            # written to both stores
    hyps = repo.list_hypotheses_for_trial(trial.trial_id)
    assert hyps and hyps[0].category == FailureCategory.likely_instability


@pytest.mark.parametrize(
    "status,exc_type,expected",
    [
        ("keep", None, TrialStatus.promoted),
        ("discard", None, TrialStatus.rejected),
        ("crash", "torch.cuda.OutOfMemoryError", TrialStatus.failed_oom),
        ("crash", "ValueError", TrialStatus.failed_runtime),
    ],
)
def test_status_mapping(make_env, status, exc_type, expected):
    settings, repo = make_env(ollama_enabled=False)
    trial = record_rejected_trial(
        {"git_commit": "abc", "status": status, "baseline_metric": 1.0},
        {"post_change_metric": 1.1 if exc_type is None else 0.0},  # 0.0 == crash sentinel
        None,
        {"exception_type": exc_type, "exception_message": "boom" if exc_type else None,
         "finished": exc_type is None},
        settings=settings, repository=repo,
    )
    assert trial.status == expected
    if exc_type:  # crash sentinel val_bpb=0.0 becomes missing, not a real 0
        assert trial.post_change_metric is None


def test_disabled_is_noop(make_env):
    settings, repo = make_env(enabled=False)
    result = record_rejected_trial(
        {"git_commit": "abc", "status": "discard"}, {"post_change_metric": 1.1}, "d", {"finished": True},
        settings=settings, repository=repo,
    )
    assert result is None
    assert repo.list_trials() == []
    assert repo.list_hypotheses() == []


# --- live hook (program.md snippet + record_from_run) ----------------------------
def test_program_md_hook_flag_guarded(make_env):
    disabled, _ = make_env(enabled=False)
    assert render_program_md_hook(disabled) is None
    enabled, _ = make_env(enabled=True)
    snippet = render_program_md_hook(enabled, branch="autoresearch/mar5")
    assert "python -m failuretrace record" in snippet
    assert "autoresearch/mar5" in snippet


def test_record_from_run_captures_diff_and_hyperparams(make_env, tmp_path):
    settings, repo = make_env(ollama_enabled=False)
    repo_root = tmp_path / "autoresearch"
    base, exp = _make_git_repo(repo_root)

    trial = record_from_run(
        settings=settings, repository=repo,
        commit=exp, status="discard", run_log_text=_RUN_LOG,
        repo_path=str(repo_root), branch="autoresearch/mar5",
        baseline_metric=1.0, baseline_commit=base, description="raise LR",
    )
    assert trial is not None
    assert trial.status == TrialStatus.rejected
    assert trial.post_change_metric == pytest.approx(1.05)
    assert trial.peak_vram_gb == pytest.approx(45060.2 / 1024)
    assert trial.code_diff and "train.py" in trial.code_diff        # git diff captured
    assert trial.hyperparameters.get("MATRIX_LR") == 0.08           # tunables parsed at commit


# --- offline batch backfill (best-effort) ---------------------------------------
def test_ingest_results_tsv_backfills_non_keep_rows(make_env, tmp_path):
    settings, repo = make_env(ollama_enabled=False)
    repo_root = tmp_path / "autoresearch"
    _make_git_repo(repo_root)
    tsv = tmp_path / "results.tsv"
    tsv.write_text(
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "a1b2c3d\t0.997900\t44.0\tkeep\tbaseline\n"
        "c3d4e5f\t1.005000\t44.0\tdiscard\tGeLU\n"
        "d4e5f6a\t0.000000\t0.0\tcrash\tOOM\n",
        encoding="utf-8",
    )
    recorded = ingest_results_tsv(settings=settings, repository=repo, tsv_path=tsv, repo_path=str(repo_root))
    assert len(recorded) == 2  # keep excluded by default
    statuses = {t.status for t in recorded}
    assert TrialStatus.rejected in statuses and TrialStatus.failed_runtime in statuses

    # idempotent: re-scanning the same TSV records nothing twice (commits already present)
    again = ingest_results_tsv(settings=settings, repository=repo, tsv_path=tsv, repo_path=str(repo_root))
    assert again == []
    assert len(repo.list_trials()) == 2


# --- optimizer adapter ----------------------------------------------------------
def test_guidance_for_produces_search_guidance(make_env):
    settings, repo = make_env(ollama_enabled=False)
    for _ in range(2):
        record_rejected_trial(
            {"git_commit": "x", "status": "discard", "baseline_metric": 1.0,
             "changed_components": ["optimizer"]},
            {"post_change_metric": 1.15},
            "d",
            {"telemetry": {"gradient_norm_mean": 1.0, "gradient_norm_std": 3.0, "val_metric": 1.15}, "finished": True},
            settings=settings, repository=repo,
        )
    ic = InterventionContext(category=FailureCategory.likely_instability, changed_components=["optimizer"])
    guidance = guidance_for(ic, settings=settings, repository=repo)
    assert guidance.soft_penalties
    terms = soft_penalty_terms(guidance)
    assert terms  # variable -> penalty flattening for a future optimizer
