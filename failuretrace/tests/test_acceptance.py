"""Executable acceptance criteria AC1-AC14 (spec §9 / Gate 6).

Criteria for later phases are ``skip``-ped with their phase and un-skipped as phases
land. This makes the acceptance list executable rather than self-graded prose. Active
now (Phase 1): AC5, AC10, AC12, AC14.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_PROVIDERS = (
    "openai",
    "anthropic",
    "google-generativeai",
    "google-genai",
    "vertexai",
    "cohere",
    "mistralai",
    "langchain",
    "llama-index",
    "wandb",
    "pinecone",
    "weaviate",
    "chromadb",
    "qdrant",
)


# --- AC1 ------------------------------------------------------------------------
@pytest.mark.skip(reason="phase 5")
def test_ac1_synthetic_rejected_trial_ingested_via_public_api():
    ...


# --- AC2 (active, phase 2) ------------------------------------------------------
def test_ac2_normalized_telemetry_record_produced():
    from failuretrace.telemetry import TelemetryRecord, normalize

    rec = normalize({"val_metric": 1.0, "peak_vram_gb": 44.0, "unknown_key": 5})
    assert isinstance(rec, TelemetryRecord)
    assert rec.val_metric == 1.0 and rec.peak_vram_gb == 44.0
    assert normalize({}).val_metric is None  # partial input accepted gracefully


# --- AC3 (active, phase 2) ------------------------------------------------------
def test_ac3_deterministic_classifier_returns_explainable_category(settings):
    from failuretrace.classifier import classify
    from failuretrace.tests.fixtures.scenarios import divergence_nan

    result = classify(divergence_nan(), settings)
    assert result.category.value == "divergence"
    assert result.observations and result.triggered_rules  # explainable


# --- AC4 (active, phase 3) ------------------------------------------------------
def test_ac4_fallback_hypothesis_persisted_when_ollama_disabled(make_env):
    from failuretrace import classify
    from failuretrace.analyst import analyze
    from failuretrace.core.enums import HypothesisSource
    from failuretrace.tests.fixtures.scenarios import instability

    settings, repo = make_env(ollama_enabled=False)
    ctx = instability()
    classification = classify(ctx, settings)
    hyp = analyze(classification, ctx, trial_id="ac4", settings=settings, repository=repo)
    assert hyp.source == HypothesisSource.rule_based
    assert repo.get_hypothesis(hyp.hypothesis_id) is not None


# --- AC5 (active) ---------------------------------------------------------------
def test_ac5_trial_written_to_both_sqlite_and_json(repo, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    assert repo.get_trial(trial.trial_id) == trial   # SQLite
    assert repo.json.exists(trial.trial_id)           # JSON
    assert repo.json.read_trial(trial.trial_id) == trial


def _seed_instability(repo, settings, make_trial):
    from failuretrace import build_fallback, classify
    from failuretrace.tests.fixtures.scenarios import instability

    trial = make_trial(changed_components=["optimizer"], hyperparameters={"MATRIX_LR": 0.08})
    repo.save_trial(trial)
    classification = classify(instability(), settings)
    hyp = build_fallback(classification, instability(), trial_id=trial.trial_id, settings=settings)
    repo.save_hypothesis(hyp)
    return hyp


# --- AC6 (active, phase 4) ------------------------------------------------------
def test_ac6_relevant_prior_failures_retrieved(repo, settings, make_trial):
    from failuretrace import InterventionContext, retrieve_relevant_failures
    from failuretrace.core.enums import FailureCategory

    _seed_instability(repo, settings, make_trial)
    ic = InterventionContext(
        category=FailureCategory.likely_instability,
        changed_components=["optimizer"],
        changed_hyperparameters={"MATRIX_LR": 0.08},
    )
    results = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    assert results
    assert results[0].hypothesis.category == FailureCategory.likely_instability
    assert results[0].relevance_score > 0


# --- AC7 (active, phase 4) ------------------------------------------------------
def test_ac7_retrieval_includes_deterministic_score_explanations(repo, settings, make_trial):
    from failuretrace import InterventionContext, retrieve_relevant_failures
    from failuretrace.core.enums import FailureCategory

    _seed_instability(repo, settings, make_trial)
    ic = InterventionContext(category=FailureCategory.likely_instability, changed_components=["optimizer"])
    results = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    assert all(rf.score_explanation for rf in results)


# --- AC8 (active, phase 4) ------------------------------------------------------
def test_ac8_counterfactual_plan_generated_without_execution(settings):
    from failuretrace import CounterfactualPlan, build_fallback, classify, plan_counterfactual
    from failuretrace.tests.fixtures.scenarios import instability

    classification = classify(instability(), settings)
    hyp = build_fallback(classification, instability(), trial_id="t", settings=settings)
    plan = plan_counterfactual(hyp, settings=settings)
    assert isinstance(plan, CounterfactualPlan)
    assert plan.treatment_variables and plan.held_constant_variables


# --- AC9 (active, phase 4) ------------------------------------------------------
def test_ac9_replication_gate_prevents_single_trial_c2_plus(settings):
    from failuretrace import ReplicationEvidence, evaluate_replication

    single = evaluate_replication(
        "h", [ReplicationEvidence(trial_id="t1", seed=42)],
        settings=settings, replication_group_id="g",
    )
    assert single is None


# --- AC10 (active) --------------------------------------------------------------
def test_ac10_metric_direction_respected():
    from failuretrace import MetricDirection, improvement

    baseline, post = 1.0, 0.9
    assert improvement(baseline, post, MetricDirection.minimize) > 0   # lower is better
    assert improvement(baseline, post, MetricDirection.maximize) < 0   # higher is better


# --- AC11 -----------------------------------------------------------------------
@pytest.mark.skip(reason="phase 5")
def test_ac11_cli_summary_and_failure_map_reports_run():
    ...


# --- AC12 (active) --------------------------------------------------------------
def test_ac12_cpu_only_offline_foundation():
    import failuretrace  # must import with no GPU / torch / network
    from failuretrace import MetricDirection, improvement

    assert improvement(1.0, 0.9, MetricDirection.minimize) > 0
    assert "torch" not in sys.modules  # the package must not pull in torch


# --- AC13 -----------------------------------------------------------------------
@pytest.mark.skip(reason="phase 5")
def test_ac13_disabled_means_autoresearch_unchanged():
    ...


# --- AC14 (active) --------------------------------------------------------------
def test_ac14_no_paid_provider_or_cloud_dependency():
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        deps.extend(extra)
    joined = " ".join(deps).lower()
    offenders = [p for p in FORBIDDEN_PROVIDERS if p in joined]
    assert not offenders, f"forbidden providers in dependencies: {offenders}"

    # none of them were imported as a side effect of importing failuretrace
    import failuretrace  # noqa: F401

    imported = [p for p in ("openai", "anthropic", "langchain", "wandb") if p in sys.modules]
    assert not imported, f"forbidden providers imported: {imported}"
