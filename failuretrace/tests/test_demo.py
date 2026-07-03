"""End-to-end demo test (Phase 6): the full pipeline including multi-seed promotion."""

from __future__ import annotations

from pathlib import Path

from failuretrace import run_demo
from failuretrace.demo import render_demo_report


def test_demo_runs_end_to_end(make_env):
    settings, repo = make_env()  # defaults (ollama on); the demo forces it off
    result = run_demo(settings, repository=repo)

    assert result.trials_ingested == 8
    assert result.category_counts.get("likely_instability") == 3

    # multi-seed replication group promoted exactly one hypothesis C1 -> C2
    assert result.promoted_hypothesis_id is not None
    assert result.promotion_supporting_trials >= 2
    assert result.effective_level_after_promotion == "C2_replicated_effect"

    # controlled counterfactual validation pushed it to C3 with a measured effect (Phase 7)
    assert result.level_after_counterfactual == "C3_counterfactual_supported"
    assert result.counterfactual_effect is not None and result.counterfactual_effect > 0
    assert result.counterfactual_n == 2

    # retrieval for a new intervention context, a counterfactual plan, and reports
    assert result.retrieval_hits > 0
    assert result.plan_id is not None
    assert result.plan_primary_variable == "MATRIX_LR"
    assert Path(result.summary_path).exists()
    assert Path(result.failure_map_path).exists()

    narrative = render_demo_report(result)
    assert "end-to-end demo" in narrative
    assert "C3_counterfactual_supported" in narrative
    assert "controlled effect" in narrative


def test_demo_forces_ollama_off_and_is_offline(make_env):
    # Even with ollama_enabled=True, the demo must not attempt any network call.
    settings, repo = make_env(ollama_enabled=True)
    result = run_demo(settings, repository=repo)
    assert result.trials_ingested == 8
    # single trials stay C0/C1; only the replicated group reached C2
    assert result.effective_level_after_promotion == "C2_replicated_effect"
