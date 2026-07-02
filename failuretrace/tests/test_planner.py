"""Planner + replication-gate tests: T10, T11, T7, T14-extended, C4."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from failuretrace import (
    CounterfactualPlan,
    CounterfactualResult,
    ReplicationEvidence,
    build_fallback,
    classify,
    evaluate_c4,
    evaluate_counterfactual,
    evaluate_replication,
    plan_counterfactual,
)
from failuretrace.core.enums import CausalSupportLevel, MetricDirection
from failuretrace.planner.interventions import DEFAULT_KNOWN_VARIABLES
from failuretrace.tests.fixtures.scenarios import (
    inconclusive_noise,
    instability,
    oom_crash,
    overfitting,
    undertraining,
)


def _hypothesis(settings, scenario_ctx, trial_id="trial_p"):
    classification = classify(scenario_ctx, settings)
    return build_fallback(classification, scenario_ctx, trial_id=trial_id, settings=settings)


# --- T10: planner holds every unrelated variable constant ------------------------
@pytest.mark.parametrize("scenario", [instability, undertraining, overfitting, oom_crash])
def test_t10_holds_unrelated_variables_constant(settings, scenario):
    hyp = _hypothesis(settings, scenario())
    plan = plan_counterfactual(hyp, settings=settings)
    assert plan is not None
    treatment = set(plan.treatment_variables)
    held = set(plan.held_constant_variables)
    # everything known and not intervened on is held constant
    assert (set(DEFAULT_KNOWN_VARIABLES) - treatment) <= held
    # intervened variables are not also "held constant"
    assert treatment.isdisjoint(held)


def test_no_plan_for_unselected_category(settings):
    hyp = _hypothesis(settings, inconclusive_noise())
    assert plan_counterfactual(hyp, settings=settings) is None


# --- T11: coupled plan requires an interaction rationale -------------------------
def test_t11_coupled_plan_without_rationale_is_rejected():
    with pytest.raises(ValidationError):
        CounterfactualPlan(
            plan_id="p", hypothesis_id="h",
            primary_intervention_variable="DEVICE_BATCH_SIZE",
            optional_coupled_stabilization_variable="grad_accum_steps",  # coupled, no rationale
            expected_outcome_if_hypothesis_true="x",
            expected_outcome_if_hypothesis_false="y",
            settings_hash="s",
        )


def test_resource_pressure_plan_is_coupled_with_rationale(settings):
    hyp = _hypothesis(settings, oom_crash())
    plan = plan_counterfactual(hyp, settings=settings)
    assert plan.optional_coupled_stabilization_variable == "grad_accum_steps"
    assert plan.interaction_rationale  # non-empty


def test_single_variable_plan_has_no_coupled_variable(settings):
    plan = plan_counterfactual(_hypothesis(settings, instability()), settings=settings)
    assert plan.optional_coupled_stabilization_variable is None


# --- T7: the gate refuses C2 from a single trial --------------------------------
def test_t7_gate_refuses_c2_from_single_trial(settings):
    single = evaluate_replication(
        "h1", [ReplicationEvidence(trial_id="t1", seed=42)],
        settings=settings, replication_group_id="g1",
    )
    assert single is None

    two = evaluate_replication(
        "h1",
        [ReplicationEvidence(trial_id="t1", seed=42), ReplicationEvidence(trial_id="t2", seed=43)],
        settings=settings, replication_group_id="g1",
    )
    assert two is not None
    assert two.to_level == CausalSupportLevel.C2_replicated_effect


# --- T14 extended: directional counterfactual result respected both ways ---------
def test_t14_counterfactual_direction_respected():
    # baseline 1.0 -> post 0.9 is an improvement under minimize, a regression under maximize.
    from failuretrace import load_settings
    settings = load_settings(env={})

    minimize = CounterfactualResult(
        trial_id="c1", baseline_metric=1.0, post_change_metric=0.9,
        metric_direction=MetricDirection.minimize,
    )
    assert evaluate_counterfactual("h", [minimize], settings=settings) is not None

    maximize = CounterfactualResult(
        trial_id="c1", baseline_metric=1.0, post_change_metric=0.9,
        metric_direction=MetricDirection.maximize,
    )
    assert evaluate_counterfactual("h", [maximize], settings=settings) is None


# --- C4 requires >=2 confirmations from >=2 distinct contexts --------------------
def test_c4_requires_two_distinct_contexts(settings):
    same_context = [
        CounterfactualResult(trial_id=f"c{i}", baseline_metric=1.0, post_change_metric=0.9,
                             metric_direction=MetricDirection.minimize,
                             changed_components=["optimizer"], config_hash="h1")
        for i in range(2)
    ]
    assert evaluate_c4("h", same_context, settings=settings) is None  # 2 results, 1 context

    distinct = [
        CounterfactualResult(trial_id="c1", baseline_metric=1.0, post_change_metric=0.9,
                             metric_direction=MetricDirection.minimize,
                             changed_components=["optimizer"], config_hash="h1"),
        CounterfactualResult(trial_id="c2", baseline_metric=1.0, post_change_metric=0.9,
                             metric_direction=MetricDirection.minimize,
                             changed_components=["data"], config_hash="h2"),
    ]
    promotion = evaluate_c4("h", distinct, settings=settings)
    assert promotion is not None
    assert promotion.to_level == CausalSupportLevel.C4_robust_rule


# --- promotion persists and raises the effective level (not the record) ----------
def test_replication_promotion_persists_and_raises_effective_level(repo, settings, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    hyp = _hypothesis(settings, instability(), trial_id=trial.trial_id)
    repo.save_hypothesis(hyp)

    promotion = evaluate_replication(
        hyp.hypothesis_id,
        [ReplicationEvidence(trial_id="a", seed=1), ReplicationEvidence(trial_id="b", seed=2)],
        settings=settings, replication_group_id="g",
    )
    repo.save_promotion(promotion)

    # the hypothesis record is unchanged; only the *effective* level moves.
    assert repo.get_hypothesis(hyp.hypothesis_id).causal_support_level == (
        CausalSupportLevel.C1_plausible_hypothesis
    )
    assert repo.effective_causal_level(hyp.hypothesis_id) == CausalSupportLevel.C2_replicated_effect
