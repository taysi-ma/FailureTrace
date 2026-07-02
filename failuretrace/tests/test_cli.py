"""CLI tests (subprocess): init -> ingest -> report, exercised as a real process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_TRIAL_JSON = {
    "git_commit": "abc1234",
    "seed": 42,
    "status": "discard",
    "baseline_metric": 1.0,
    "post_change_metric": 1.15,
    "changed_components": ["optimizer"],
    "hyperparameters": {"MATRIX_LR": 0.08},
    "telemetry": {"gradient_norm_mean": 1.0, "gradient_norm_std": 3.0, "val_metric": 1.15},
    "finished": True,
}


def _run(tmp_path: Path, *cli_args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(_REPO_ROOT))
    globals_ = ["--data-dir", str(tmp_path / "data"), "--reports-dir", str(tmp_path / "reports"), "--no-ollama"]
    return subprocess.run(
        [sys.executable, "-m", "failuretrace", *globals_, *cli_args],
        capture_output=True, text=True, cwd=str(_REPO_ROOT), env=env, timeout=120,
    )


def test_cli_init(tmp_path):
    result = _run(tmp_path, "init")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "data").exists()


def test_cli_ingest_then_reports(tmp_path):
    assert _run(tmp_path, "init").returncode == 0

    trial_file = tmp_path / "trial.json"
    trial_file.write_text(json.dumps(_TRIAL_JSON), encoding="utf-8")
    ingest = _run(tmp_path, "ingest", str(trial_file))
    assert ingest.returncode == 0, ingest.stderr
    assert "ingested trial" in ingest.stdout

    # summary + map reports produce artifacts
    summary = _run(tmp_path, "report", "summary")
    assert summary.returncode == 0, summary.stderr
    assert (tmp_path / "reports" / "summary.md").exists()

    failures = _run(tmp_path, "report", "failures")
    assert failures.returncode == 0, failures.stderr

    mp = _run(tmp_path, "report", "map")
    assert mp.returncode == 0, mp.stderr
    assert (tmp_path / "reports" / "failure_map.md").exists()


def test_cli_gate_and_guidance(tmp_path):
    assert _run(tmp_path, "init").returncode == 0
    # ingest three same-family instability trials on distinct commits -> a replication group
    for i in range(3):
        trial_file = tmp_path / f"t{i}.json"
        trial_file.write_text(json.dumps({
            **_TRIAL_JSON, "git_commit": f"commit{i}", "seed": 40 + i,
        }), encoding="utf-8")
        assert _run(tmp_path, "ingest", str(trial_file)).returncode == 0

    gate = _run(tmp_path, "gate")
    assert gate.returncode == 0, gate.stderr
    assert "C1 -> C2" in gate.stdout  # replication promotion reported

    # gate is idempotent on a second run
    gate2 = _run(tmp_path, "gate")
    assert "no hypothesis met" in gate2.stdout

    guidance = _run(tmp_path, "guidance", "--category", "likely_instability", "--component", "optimizer")
    assert guidance.returncode == 0, guidance.stderr
    assert "hard constraint" in guidance.stdout  # C2 evidence -> hard constraint


def test_cli_report_trial(tmp_path):
    assert _run(tmp_path, "init").returncode == 0
    trial_file = tmp_path / "trial.json"
    trial_file.write_text(json.dumps(_TRIAL_JSON), encoding="utf-8")
    ingest = _run(tmp_path, "ingest", str(trial_file))
    trial_id = ingest.stdout.split("ingested trial", 1)[1].split()[0]

    report = _run(tmp_path, "report", "trial", trial_id)
    assert report.returncode == 0, report.stderr
    assert (tmp_path / "reports" / f"trial_{trial_id}.md").exists()
    assert "likely_instability" in report.stdout
