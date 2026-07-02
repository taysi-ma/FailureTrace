"""Reporting tests: summary (level separation + non-causal headers), failure map, trial."""

from __future__ import annotations

from failuretrace import record_rejected_trial
from failuretrace.reporting import (
    build_failure_map,
    build_summary,
    render_failure_map_text,
    render_summary_text,
    write_failure_map,
    write_summary,
    write_trial_report,
)


def _seed(settings, repo):
    # instability (rejected, C1)
    inst = record_rejected_trial(
        {"git_commit": "aaa", "status": "discard", "baseline_metric": 1.0, "changed_components": ["optimizer"]},
        {"post_change_metric": 1.15},
        "d", {"telemetry": {"gradient_norm_mean": 1.0, "gradient_norm_std": 3.0, "val_metric": 1.15}, "finished": True},
        settings=settings, repository=repo,
    )
    # OOM crash (failed_oom -> resource_pressure)
    record_rejected_trial(
        {"git_commit": "bbb", "status": "crash", "baseline_metric": 1.0},
        {"post_change_metric": 0.0},
        None, {"exception_type": "torch.cuda.OutOfMemoryError", "exception_message": "CUDA out of memory", "finished": False},
        settings=settings, repository=repo,
    )
    return inst


def test_build_and_render_summary_separates_levels(make_env):
    settings, repo = make_env(ollama_enabled=False)
    _seed(settings, repo)

    summary = build_summary(repo, settings)
    assert summary.total_trials == 2
    assert summary.total_hypotheses == 2
    assert "rejected" in summary.status_counts and "failed_oom" in summary.status_counts

    text = render_summary_text(summary)
    # explicit non-causal headers for C0/C1
    assert "Plausible hypotheses (C1 — NOT causally validated)" in text
    assert "Observations (C0" in text
    assert "Replicated effects (C2" in text
    assert "Counterfactual-supported effects (C3)" in text
    assert "Robust rules (C4" in text
    assert "not** causal" in text.lower() or "NOT causal" in text  # caveat present


def test_write_summary_creates_artifact(make_env):
    settings, repo = make_env(ollama_enabled=False)
    _seed(settings, repo)
    path = write_summary(repo, settings)
    assert path.exists() and path.name == "summary.md"
    assert "governance summary" in path.read_text(encoding="utf-8")


def test_failure_map_build_render_write(make_env):
    settings, repo = make_env(ollama_enabled=False)
    _seed(settings, repo)
    rows = build_failure_map(repo, settings)
    assert rows
    categories = {r.category for r in rows}
    assert "likely_instability" in categories and "resource_pressure" in categories

    text = render_failure_map_text(rows)
    assert "| category |" in text
    path = write_failure_map(repo, settings)
    assert path.exists() and path.name == "failure_map.md"


def test_trial_report_written(make_env):
    settings, repo = make_env(ollama_enabled=False)
    trial = _seed(settings, repo)
    path = write_trial_report(repo, settings, trial.trial_id)
    content = path.read_text(encoding="utf-8")
    assert path.name == f"trial_{trial.trial_id}.md"
    assert "likely_instability" in content
    assert "direction-aware improvement" in content


def test_empty_reports_do_not_crash(make_env):
    settings, repo = make_env(ollama_enabled=False)
    summary = build_summary(repo, settings)
    assert summary.total_trials == 0
    assert render_summary_text(summary)          # renders with all sections "(none)"
    assert "No failure hypotheses" in render_failure_map_text(build_failure_map(repo, settings))
