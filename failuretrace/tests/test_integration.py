"""Integration adapter tests: public API, live hook, offline backfill, optimizer feed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from failuretrace import (
    InterventionContext,
    components_for,
    infer_changed_tunables,
    record_from_run,
    record_rejected_trial,
    render_program_md_consult_hook,
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

# Realistic autoresearch crash artifacts (the exact stdout shapes train.py produces on an
# OOM traceback and on the NaN/loss-blowup FAIL marker). Used to validate the live adapter
# end-to-end on CPU — a real GPU run is out of the provider-free remit (see docs).
_OOM_RUN_LOG = """\
step 00007 (0.7%) | loss: 3.21 | lrm: 0.04 | dt: 310ms | tok/sec: 1.6e6 | mfu: 39.8 | epoch: 0
Traceback (most recent call last):
  File "train.py", line 552, in <module>
    loss.backward()
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
"""
_NAN_RUN_LOG = """\
step 00010 (1.0%) | loss: nan | lrm: 0.09 | dt: 300ms | tok/sec: 1.6e6 | mfu: 39.8 | epoch: 0
FAIL
"""


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
    assert trial.telemetry["train_loss_start"] == pytest.approx(2.10)
    assert trial.telemetry["train_loss_end"] == pytest.approx(2.10)
    assert trial.telemetry["learning_rate_history"] == [0.02]
    assert trial.telemetry["step_throughput_mean"] == pytest.approx(1.6e6)
    assert trial.telemetry["step_mfu_mean"] == pytest.approx(39.8)
    assert trial.telemetry["num_steps"] == 953
    assert trial.telemetry["num_params_m"] == pytest.approx(50.3)
    assert trial.telemetry["depth"] == 8


def test_record_from_run_sets_config_hash_and_parent_lineage(make_env, tmp_path):
    settings, repo = make_env(ollama_enabled=False)
    repo_root = tmp_path / "autoresearch"
    base, exp = _make_git_repo(repo_root)

    baseline = record_from_run(
        settings=settings, repository=repo, commit=base, status="discard",
        run_log_text=_RUN_LOG, repo_path=str(repo_root), baseline_metric=1.0,
    )
    experiment = record_from_run(
        settings=settings, repository=repo, commit=exp, status="discard",
        run_log_text=_RUN_LOG, repo_path=str(repo_root), baseline_metric=1.0, baseline_commit=base,
    )
    # config identity from the tunable block, and parent linked to the recorded baseline (#6)
    assert experiment.config_hash is not None
    assert experiment.config_hash != baseline.config_hash  # MATRIX_LR differs
    assert experiment.parent_trial_id == baseline.trial_id


def test_collect_telemetry_false_ingests_without_telemetry(make_env):
    settings, repo = make_env(ollama_enabled=False, collect_telemetry=False)
    trial = record_rejected_trial(
        {"git_commit": "x", "status": "discard", "baseline_metric": 1.0, "changed_components": ["optimizer"]},
        {"post_change_metric": 1.15},
        "d",
        {"telemetry": {"gradient_norm_mean": 1.0, "gradient_norm_std": 3.0, "val_metric": 1.15}, "finished": True},
        settings=settings, repository=repo,
    )
    assert trial.telemetry.get("gradient_norm_mean") is None  # telemetry not collected
    # with no telemetry the instability rule cannot fire -> degrades, never crashes
    assert repo.list_hypotheses_for_trial(trial.trial_id)[0].category != FailureCategory.likely_instability


def test_ingestion_auto_plans_counterfactual_for_plannable_category(make_env):
    settings, repo = make_env(ollama_enabled=False)  # counterfactual_planner_enabled default true
    trial = record_rejected_trial(
        {"git_commit": "x", "status": "discard", "baseline_metric": 1.0, "changed_components": ["optimizer"],
         "hyperparameters": {"MATRIX_LR": 0.08}, "changed_hyperparameters": {"MATRIX_LR": 0.08}},
        {"post_change_metric": 1.15},
        "d",
        {"telemetry": {"gradient_norm_mean": 1.0, "gradient_norm_std": 3.0, "val_metric": 1.15}, "finished": True},
        settings=settings, repository=repo,
    )
    hyp = repo.list_hypotheses_for_trial(trial.trial_id)[0]
    plans = repo.list_plans_for_hypothesis(hyp.hypothesis_id)
    assert plans and plans[0].primary_intervention_variable == "MATRIX_LR"


def test_ingestion_persists_deterministic_classification(make_env):
    settings, repo = make_env(ollama_enabled=False)
    trial = record_rejected_trial(
        {"git_commit": "x", "status": "discard", "baseline_metric": 1.0, "changed_components": ["optimizer"]},
        {"post_change_metric": 1.15},
        "d",
        {"telemetry": {"gradient_norm_mean": 1.0, "gradient_norm_std": 3.0, "val_metric": 1.15}, "finished": True},
        settings=settings, repository=repo,
    )
    classification = repo.get_classification(trial.trial_id)
    assert classification is not None
    assert classification.category == FailureCategory.likely_instability
    assert classification.triggered_rules  # deterministic provenance preserved verbatim


# --- realistic autoresearch artifacts, end-to-end (CPU-only stand-in for a GPU run) ---
def test_record_from_run_oom_traceback_is_resource_pressure(make_env, tmp_path):
    settings, repo = make_env(ollama_enabled=False)
    repo_root = tmp_path / "autoresearch"
    base, exp = _make_git_repo(repo_root)
    trial = record_from_run(
        settings=settings, repository=repo, commit=exp, status="crash",
        run_log_text=_OOM_RUN_LOG, repo_path=str(repo_root), baseline_metric=1.0, baseline_commit=base,
    )
    assert trial.status == TrialStatus.failed_oom
    assert repo.list_hypotheses_for_trial(trial.trial_id)[0].category == FailureCategory.resource_pressure
    assert repo.get_classification(trial.trial_id).category == FailureCategory.resource_pressure


def test_record_from_run_nan_fail_marker_is_divergence(make_env, tmp_path):
    settings, repo = make_env(ollama_enabled=False)
    repo_root = tmp_path / "autoresearch"
    _base, exp = _make_git_repo(repo_root)
    trial = record_from_run(
        settings=settings, repository=repo, commit=exp, status="crash",
        run_log_text=_NAN_RUN_LOG, repo_path=str(repo_root), baseline_metric=1.0,
    )
    assert repo.list_hypotheses_for_trial(trial.trial_id)[0].category == FailureCategory.divergence


def test_realistic_results_tsv_corpus_offline(make_env, tmp_path):
    settings, repo = make_env(ollama_enabled=False)
    repo_root = tmp_path / "autoresearch"
    _make_git_repo(repo_root)
    # a realistic multi-row corpus in the exact program.md contract (keep/discard/crash,
    # val_bpb=0.000000 + memory_gb=0.0 crash sentinels, commas-free descriptions)
    tsv = tmp_path / "results.tsv"
    tsv.write_text(
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "a1b2c3d\t0.997900\t44.0\tkeep\tbaseline\n"
        "c3d4e5f\t1.005000\t44.0\tdiscard\tswitch to GeLU activation\n"
        "d4e5f6a\t0.000000\t0.0\tcrash\tdouble model width OOM\n"
        "e5f6a7b\t1.002000\t44.1\tdiscard\traise dropout\n",
        encoding="utf-8",
    )
    recorded = ingest_results_tsv(settings=settings, repository=repo, tsv_path=tsv, repo_path=str(repo_root))
    assert len(recorded) == 3  # keep excluded by default
    # crash row: the val_bpb=0.0 sentinel is treated as missing, never a real 0 bpb
    crash = next(t for t in recorded if t.status in (TrialStatus.failed_oom, TrialStatus.failed_runtime))
    assert crash.post_change_metric is None
    # every row produced a persisted classification (nothing silently dropped), idempotent re-scan
    assert all(repo.get_classification(t.trial_id) is not None for t in recorded)
    assert ingest_results_tsv(settings=settings, repository=repo, tsv_path=tsv, repo_path=str(repo_root)) == []


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


# --- Phase 8: pre-experiment inference + the consult hook -----------------------
def _make_repo_with_uncommitted_edit(root: Path) -> None:
    """A committed baseline plus an *uncommitted* train.py edit — the state the agent is
    in when it should consult the brief (edited, not yet committed)."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    (root / "train.py").write_text(_TRAIN_PY_BASE, encoding="utf-8")
    _git(root, "add", "train.py")
    _git(root, "commit", "-q", "-m", "baseline")
    (root / "train.py").write_text(_TRAIN_PY_EXP, encoding="utf-8")


def test_infer_changed_tunables_detects_working_tree_edit(tmp_path):
    root = tmp_path / "autoresearch"
    _make_repo_with_uncommitted_edit(root)
    changed = infer_changed_tunables(root)
    # only the knob that actually moved; unchanged knobs must not be reported
    assert changed == {"MATRIX_LR": 0.08}


def test_infer_changed_tunables_is_empty_without_git_or_file(tmp_path):
    assert infer_changed_tunables(tmp_path / "nonexistent") == {}
    bare = tmp_path / "no_git"
    bare.mkdir()
    (bare / "train.py").write_text(_TRAIN_PY_EXP, encoding="utf-8")
    assert infer_changed_tunables(bare) == {}  # no ref to diff against => no guess


def test_components_for_maps_tunables_via_config(make_env):
    settings, _ = make_env(ollama_enabled=False)
    assert components_for(["MATRIX_LR"], settings) == ["optimizer"]
    assert components_for(["DEVICE_BATCH_SIZE", "DEPTH"], settings) == ["batch", "model"]
    assert components_for(["NOT_A_KNOB"], settings) == []


def test_consult_hook_renders_when_enabled(make_env):
    settings, _ = make_env(ollama_enabled=False)
    hook = render_program_md_consult_hook(settings, repo=".")
    assert hook is not None
    assert "failuretrace brief --infer-from ." in hook
    # the hook must state the epistemic contract, not just the command
    assert "NOT causally validated" in hook


def test_consult_hook_absent_when_disabled(make_env):
    settings, _ = make_env(ollama_enabled=False, enabled=False)
    assert render_program_md_consult_hook(settings) is None
    assert render_program_md_hook(settings) is None
