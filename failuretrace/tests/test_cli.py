"""CLI tests (subprocess): init -> ingest -> report, exercised as a real process."""

from __future__ import annotations

import json
import os
import re
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


def test_cli_brief_formats_and_epistemic_labelling(tmp_path):
    """A single unreplicated failure must reach the agent as context, not as a rule."""
    assert _run(tmp_path, "init").returncode == 0
    trial_file = tmp_path / "trial.json"
    trial_file.write_text(json.dumps(_TRIAL_JSON), encoding="utf-8")
    assert _run(tmp_path, "ingest", str(trial_file)).returncode == 0

    md = _run(tmp_path, "brief", "--component", "optimizer", "--param", "MATRIX_LR=0.08")
    assert md.returncode == 0, md.stderr
    assert "NOT causally validated" in md.stdout
    assert "Binding constraints" not in md.stdout  # one C1 trial earns no binding rule

    text = _run(tmp_path, "brief", "--component", "optimizer", "--format", "text")
    assert text.returncode == 0, text.stderr
    assert "plausible" in text.stdout

    js = _run(tmp_path, "brief", "--component", "optimizer", "--format", "json")
    assert js.returncode == 0, js.stderr
    payload = json.loads(js.stdout)
    assert payload["hard_constraints"] == []
    assert payload["plausible_context"]


def test_cli_brief_requires_some_context(tmp_path):
    assert _run(tmp_path, "init").returncode == 0
    result = _run(tmp_path, "brief")
    assert result.returncode == 2
    assert "no experiment context given" in result.stdout


def test_cli_brief_reports_no_relevant_failures_on_empty_store(tmp_path):
    assert _run(tmp_path, "init").returncode == 0
    result = _run(tmp_path, "brief", "--component", "optimizer")
    assert result.returncode == 0, result.stderr
    assert "No relevant prior failures" in result.stdout


def test_cli_brief_surfaces_binding_constraint_after_replication(tmp_path):
    assert _run(tmp_path, "init").returncode == 0
    for i in range(3):
        trial_file = tmp_path / f"t{i}.json"
        trial_file.write_text(json.dumps({
            **_TRIAL_JSON, "git_commit": f"commit{i}", "seed": 40 + i,
        }), encoding="utf-8")
        assert _run(tmp_path, "ingest", str(trial_file)).returncode == 0
    assert _run(tmp_path, "gate").returncode == 0  # C1 -> C2

    brief = _run(tmp_path, "brief", "--component", "optimizer", "--category", "likely_instability")
    assert brief.returncode == 0, brief.stderr
    assert "Binding constraints" in brief.stdout


def _top_score(result: subprocess.CompletedProcess) -> float:
    context = json.loads(result.stdout)["plausible_context"]
    assert context, result.stdout
    return float(re.search(r"score=([\d.]+)", context[0]).group(1))


def test_cli_param_affects_ranking(tmp_path):
    """--param must actually reach retrieval.

    hyperparameter_overlap (1.5) and range_proximity (1.0) are 2 of the 7 scoring
    components and were unreachable from the CLI before Phase 8, so a matching --param
    has to raise the relevance score above the no-param baseline.
    """
    assert _run(tmp_path, "init").returncode == 0
    trial_file = tmp_path / "trial.json"
    trial_file.write_text(json.dumps({
        **_TRIAL_JSON, "hyperparameters": {"MATRIX_LR": 0.08},
    }), encoding="utf-8")
    assert _run(tmp_path, "ingest", str(trial_file)).returncode == 0

    base = _top_score(_run(tmp_path, "brief", "--component", "optimizer", "--format", "json"))
    near = _top_score(_run(
        tmp_path, "brief", "--component", "optimizer",
        "--param", "MATRIX_LR=0.079", "--format", "json",
    ))
    far = _top_score(_run(
        tmp_path, "brief", "--component", "optimizer",
        "--param", "MATRIX_LR=0.0001", "--format", "json",
    ))
    assert near > base          # a matching hyperparameter contributes to the score
    assert near > far           # and a closer value contributes more (log-space proximity)


def test_cli_brief_is_noop_when_disabled(tmp_path):
    """The flag must hold at the CLI boundary too, not just in the library."""
    from failuretrace.core.settings import default_config_path

    config = tmp_path / "disabled.yaml"
    config.write_text(
        default_config_path().read_text(encoding="utf-8").replace("enabled: true", "enabled: false", 1),
        encoding="utf-8",
    )
    result = _run(tmp_path, "--config", str(config), "brief", "--component", "optimizer")
    assert result.returncode == 0, result.stderr
    assert "disabled" in result.stdout
    assert not (tmp_path / "data" / "failuretrace.db").exists()  # zero writes


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
