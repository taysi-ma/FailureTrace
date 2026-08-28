"""CPU-only checks for the external Lightning/Colab compatibility tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.lightning import patch_t4, patch_train, run_trials


def test_sdpa_patch_is_idempotent(tmp_path: Path) -> None:
    train_py = tmp_path / "train.py"
    train_py.write_text(f"import torch\n{patch_train._OLD}\n", encoding="utf-8")

    assert patch_train.patch_file(train_py) is True
    once = train_py.read_text(encoding="utf-8")
    assert "_ft_sdpa_flash_attn_func" in once
    assert patch_train.patch_file(train_py) is False
    assert train_py.read_text(encoding="utf-8") == once


def test_t4_profile_shape_precision_and_idempotence(tmp_path: Path) -> None:
    repo = tmp_path / "autoresearch"
    repo.mkdir()
    train_py = repo / "train.py"
    prepare_py = repo / "prepare.py"
    train_py.write_text(
        "\n".join(old for old, _new in patch_t4._TRAIN_REPLACEMENTS) + "\n",
        encoding="utf-8",
    )
    prepare_py.write_text(
        "\n".join(old for old, _new in patch_t4._PREPARE_REPLACEMENTS) + "\n",
        encoding="utf-8",
    )

    assert patch_t4.patch_repo(repo) is True
    patched_train = train_py.read_text(encoding="utf-8")
    patched_prepare = prepare_py.read_text(encoding="utf-8")
    assert "DEPTH = 8" in patched_train
    assert "DEVICE_BATCH_SIZE = 8" in patched_train
    assert "TOTAL_BATCH_SIZE = 2**14" in patched_train
    assert 'WINDOW_PATTERN = "L"' in patched_train
    # fp32 throughout: this trainer assumes bf16's exponent range and ships no GradScaler,
    # so fp16 overflows in the ReLU-squared MLP, Newton-Schulz, and the fused AdamW.
    assert "torch.bfloat16" not in patched_train
    assert "torch.float16" not in patched_train
    assert ".half()" not in patched_train
    assert "enabled=False" in patched_train
    assert "MAX_SEQ_LEN = 1024" in patched_prepare
    assert patch_t4.patch_repo(repo) is False
    assert train_py.read_text(encoding="utf-8") == patched_train
    assert prepare_py.read_text(encoding="utf-8") == patched_prepare


def test_run_config_applies_real_seed_and_restores_source(tmp_path: Path) -> None:
    train_py = tmp_path / "train.py"
    original = (
        "DEVICE_BATCH_SIZE = 8\n"
        "torch.manual_seed(42)\n"
        "torch.cuda.manual_seed(42)\n"
    )
    train_py.write_text(original, encoding="utf-8")

    saved = run_trials._set_run_config(train_py, {"DEVICE_BATCH_SIZE": 16}, seed=77)
    changed = train_py.read_text(encoding="utf-8")
    assert saved == original
    assert "DEVICE_BATCH_SIZE = 16" in changed
    assert "torch.manual_seed(77)" in changed
    assert "torch.cuda.manual_seed(77)" in changed


def test_source_fingerprint_ignores_seed_but_not_configuration(tmp_path: Path) -> None:
    repo = tmp_path / "autoresearch"
    repo.mkdir()
    train_py = repo / "train.py"
    prepare_py = repo / "prepare.py"
    prepare_py.write_text("MAX_SEQ_LEN = 512\n", encoding="utf-8")
    train_py.write_text(
        "DEVICE_BATCH_SIZE = 8\ntorch.manual_seed(42)\ntorch.cuda.manual_seed(42)\n",
        encoding="utf-8",
    )
    first = run_trials._source_fingerprint(repo)
    train_py.write_text(
        "DEVICE_BATCH_SIZE = 8\ntorch.manual_seed(43)\ntorch.cuda.manual_seed(43)\n",
        encoding="utf-8",
    )
    assert run_trials._source_fingerprint(repo) == first
    train_py.write_text(
        "DEVICE_BATCH_SIZE = 16\ntorch.manual_seed(43)\ntorch.cuda.manual_seed(43)\n",
        encoding="utf-8",
    )
    assert run_trials._source_fingerprint(repo) != first


def test_failed_baseline_is_rejected_before_ingestion() -> None:
    artifact = run_trials.RunArtifact(
        log_text="torch.OutOfMemoryError: CUDA out of memory\n",
        returncode=1,
        config_hash="cfg",
        code_diff="",
    )
    with pytest.raises(SystemExit, match="aborting before any failure evidence"):
        run_trials._require_baseline({"label": "baseline"}, artifact)


def test_failed_counterfactual_is_not_persisted_as_completed() -> None:
    artifact = run_trials.RunArtifact(
        log_text="torch.OutOfMemoryError: CUDA out of memory\n",
        returncode=1,
        config_hash="cfg",
        code_diff="",
    )
    with pytest.raises(SystemExit, match="refusing to persist it as completed"):
        run_trials._save_counterfactual(
            repo=None,
            settings=None,
            cfg={"label": "fix"},
            artifact=artifact,
            baseline=1.0,
            git_commit="abc123",
        )


def test_t4_trial_profile_uses_distinct_real_seeds() -> None:
    assert [cfg["seed"] for cfg in run_trials.T4_CONFIGS] == [42, 43, 44, 45]
    assert run_trials.T4_CONFIGS[1]["overrides"] == run_trials.T4_CONFIGS[2]["overrides"]


def test_every_trial_config_satisfies_the_grad_accum_assertion() -> None:
    """train.py:513 asserts TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0.

    A config that violates it dies on an AssertionError before it can OOM, producing a
    `runtime_failure` where the design needs `resource_pressure`. Both built-in config
    sets shipped broken until a real Kaggle T4 run surfaced it; this test catches the
    next one without spending a GPU-hour.
    """
    import re

    def _patched_int(replacements, name: str) -> int:
        """The value patch_t4 writes for `name`, e.g. MAX_SEQ_LEN -> 1024."""
        for _old, new in replacements:
            match = re.match(rf"{name} = (\S+)", new)
            if match:
                return int(eval(match.group(1)))  # noqa: S307 - literals from our own file
        raise AssertionError(f"{name} not set by the T4 profile")

    # T4 profile values come from the patch itself, so this test follows any retune.
    t4 = {
        "MAX_SEQ_LEN": _patched_int(patch_t4._PREPARE_REPLACEMENTS, "MAX_SEQ_LEN"),
        "DEVICE_BATCH_SIZE": _patched_int(patch_t4._TRAIN_REPLACEMENTS, "DEVICE_BATCH_SIZE"),
        "TOTAL_BATCH_SIZE": _patched_int(patch_t4._TRAIN_REPLACEMENTS, "TOTAL_BATCH_SIZE"),
    }
    # A100 runs unpatched upstream values (train.py:449,451 / prepare.py:30 at 228791f).
    a100 = {"MAX_SEQ_LEN": 2048, "DEVICE_BATCH_SIZE": 128, "TOTAL_BATCH_SIZE": 2**19}

    for profile, defaults, configs in (
        ("a100", a100, run_trials.CONFIGS),
        ("t4", t4, run_trials.T4_CONFIGS),
    ):
        for cfg in configs:
            overrides = cfg["overrides"]
            device_batch = overrides.get("DEVICE_BATCH_SIZE", defaults["DEVICE_BATCH_SIZE"])
            total_batch = overrides.get("TOTAL_BATCH_SIZE", defaults["TOTAL_BATCH_SIZE"])
            tokens_per_fwdbwd = device_batch * defaults["MAX_SEQ_LEN"]
            assert total_batch % tokens_per_fwdbwd == 0, (
                f"{profile}/{cfg['label']}: TOTAL_BATCH_SIZE={total_batch} is not a multiple "
                f"of DEVICE_BATCH_SIZE*MAX_SEQ_LEN={tokens_per_fwdbwd}; train.py:513 would "
                f"raise AssertionError instead of running"
            )
