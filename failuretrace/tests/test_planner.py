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
    promote_replications,
)
from failuretrace.core.enums import CausalSupportLevel, LinkType, MetricDirection
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


# --- helpers that persist real trials/hypotheses and walk the promotion ladder ----
def _instability_trial(make_trial, *, trial_id, seed):
    return make_trial(
        trial_id=trial_id, seed=seed, changed_components=["optimizer"],
        hyperparameters={"MATRIX_LR": 0.08}, baseline_metric=1.0, post_change_metric=1.15,
    )


def _persist_c1(repo, settings, make_trial, *, trial_id="src", seed=42):
    trial = _instability_trial(make_trial, trial_id=trial_id, seed=seed)
    repo.save_trial(trial)
    hyp = build_fallback(classify(instability(), settings), instability(),
                         trial_id=trial.trial_id, settings=settings)
    repo.save_hypothesis(hyp)
    return trial, hyp


def _replicate(repo, settings, make_trial, hyp, *, seeds=(1, 2)):
    evidence = []
    for s in seeds:
        t = _instability_trial(make_trial, trial_id=f"rep{s}", seed=s)
        repo.save_trial(t)
        evidence.append(ReplicationEvidence(trial_id=t.trial_id, seed=s))
    return evaluate_replication(
        hyp.hypothesis_id, evidence, settings=settings, repository=repo, replication_group_id="g",
    )


def _promote_c2(repo, settings, make_trial, hyp):
    repo.save_promotion(_replicate(repo, settings, make_trial, hyp))


def _promote_c3(repo, settings, make_trial, hyp):
    repo.save_plan(plan_counterfactual(hyp, settings=settings))
    cf = make_trial(trial_id="cf", baseline_metric=1.0, post_change_metric=0.9)
    repo.save_trial(cf)
    res = CounterfactualResult(trial_id="cf", baseline_metric=1.0, post_change_metric=0.9,
                               metric_direction=MetricDirection.minimize)
    repo.save_promotion(evaluate_counterfactual(hyp.hypothesis_id, [res], settings=settings, repository=repo))


def _persist_instability_group(repo, settings, make_trial, *, seeds, commits):
    """Persist several C1 instability hypotheses sharing one intervention fingerprint."""
    for i, (seed, commit) in enumerate(zip(seeds, commits)):
        t = make_trial(trial_id=f"grp{i}", seed=seed, git_commit=commit,
                       changed_components=["optimizer"], hyperparameters={"MATRIX_LR": 0.08},
                       baseline_metric=1.0, post_change_metric=1.15)
        repo.save_trial(t)
        repo.save_hypothesis(build_fallback(classify(instability(), settings), instability(),
                                            trial_id=t.trial_id, settings=settings))


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
def test_t7_gate_refuses_c2_from_single_trial(repo, settings, make_trial):
    _, hyp = _persist_c1(repo, settings, make_trial)

    one = _instability_trial(make_trial, trial_id="one", seed=42)
    repo.save_trial(one)
    single = evaluate_replication(
        hyp.hypothesis_id, [ReplicationEvidence(trial_id="one", seed=42)],
        settings=settings, repository=repo, replication_group_id="g1",
    )
    assert single is None  # one seed < replication minimum

    two = _replicate(repo, settings, make_trial, hyp)  # two distinct-seed real trials
    assert two is not None
    assert two.to_level == CausalSupportLevel.C2_replicated_effect


def test_replication_ignores_fabricated_and_mismatched_evidence(repo, settings, make_trial):
    _, hyp = _persist_c1(repo, settings, make_trial)
    # trials that do not exist are ignored; a different-family trial does not count.
    other = make_trial(trial_id="other", seed=7, changed_components=["data"],
                       hyperparameters={"WEIGHT_DECAY": 0.3}, baseline_metric=1.0, post_change_metric=1.15)
    repo.save_trial(other)
    result = evaluate_replication(
        hyp.hypothesis_id,
        [ReplicationEvidence(trial_id="ghost", seed=1), ReplicationEvidence(trial_id="other", seed=7)],
        settings=settings, repository=repo, replication_group_id="g",
    )
    assert result is None  # one ghost + one wrong-family => no qualifying replication


# --- T14 extended: directional counterfactual result respected both ways ---------
def test_t14_counterfactual_direction_respected(repo, settings, make_trial):
    # baseline 1.0 -> post 0.9 is an improvement under minimize, a regression under maximize.
    _, hyp = _persist_c1(repo, settings, make_trial)
    _promote_c2(repo, settings, make_trial, hyp)  # effective level now C2
    repo.save_plan(plan_counterfactual(hyp, settings=settings))  # C3 requires a persisted plan

    minimize = CounterfactualResult(
        trial_id="c1", baseline_metric=1.0, post_change_metric=0.9,
        metric_direction=MetricDirection.minimize,
    )
    assert evaluate_counterfactual(hyp.hypothesis_id, [minimize], settings=settings, repository=repo) is not None

    maximize = CounterfactualResult(
        trial_id="c1", baseline_metric=1.0, post_change_metric=0.9,
        metric_direction=MetricDirection.maximize,
    )
    assert evaluate_counterfactual(hyp.hypothesis_id, [maximize], settings=settings, repository=repo) is None


def test_counterfactual_requires_c2_and_a_persisted_plan(repo, settings, make_trial):
    _, hyp = _persist_c1(repo, settings, make_trial)  # only C1, no plan yet
    result = CounterfactualResult(trial_id="c1", baseline_metric=1.0, post_change_metric=0.9,
                                  metric_direction=MetricDirection.minimize)
    # not yet C2 -> no C3
    assert evaluate_counterfactual(hyp.hypothesis_id, [result], settings=settings, repository=repo) is None
    _promote_c2(repo, settings, make_trial, hyp)  # now C2 but still no plan persisted
    assert evaluate_counterfactual(hyp.hypothesis_id, [result], settings=settings, repository=repo) is None


# --- C4 requires >=2 confirmations from >=2 distinct contexts --------------------
def test_c4_requires_two_distinct_contexts(repo, settings, make_trial):
    _, hyp = _persist_c1(repo, settings, make_trial)
    _promote_c2(repo, settings, make_trial, hyp)
    _promote_c3(repo, settings, make_trial, hyp)  # effective level now C3

    same_context = [
        CounterfactualResult(trial_id=f"c{i}", baseline_metric=1.0, post_change_metric=0.9,
                             metric_direction=MetricDirection.minimize,
                             changed_components=["optimizer"], config_hash="h1")
        for i in range(2)
    ]
    assert evaluate_c4(hyp.hypothesis_id, same_context, settings=settings, repository=repo) is None

    distinct = [
        CounterfactualResult(trial_id="c1", baseline_metric=1.0, post_change_metric=0.9,
                             metric_direction=MetricDirection.minimize,
                             changed_components=["optimizer"], config_hash="h1"),
        CounterfactualResult(trial_id="c2", baseline_metric=1.0, post_change_metric=0.9,
                             metric_direction=MetricDirection.minimize,
                             changed_components=["data"], config_hash="h2"),
    ]
    promotion = evaluate_c4(hyp.hypothesis_id, distinct, settings=settings, repository=repo)
    assert promotion is not None
    assert promotion.to_level == CausalSupportLevel.C4_robust_rule


# --- gate driver (promote_replications) -----------------------------------------
def test_promote_replications_promotes_group_and_writes_links(repo, settings, make_trial):
    _persist_instability_group(repo, settings, make_trial, seeds=(1, 2, 3),
                               commits=["c1", "c2", "c3"])
    promotions = promote_replications(repo, settings)
    assert len(promotions) == 1
    promo = promotions[0]
    assert promo.to_level == CausalSupportLevel.C2_replicated_effect
    assert repo.effective_causal_level(promo.hypothesis_id) == CausalSupportLevel.C2_replicated_effect
    # explicit append-only replication links were written for the supporting trials
    links = repo.list_links_for_hypothesis(promo.hypothesis_id)
    assert links and all(link.link_type == LinkType.replication for link in links)
    # idempotent: the group is already represented, so a second run promotes nothing
    assert promote_replications(repo, settings) == []


def test_replication_key_uses_seed_commit_units_not_just_seed(repo, settings, make_trial):
    # autoresearch pins seed 42; three distinct COMMITS at the same seed must still replicate
    _persist_instability_group(repo, settings, make_trial, seeds=(42, 42, 42),
                               commits=["a", "b", "c"])
    assert len(promote_replications(repo, settings)) == 1


def test_deterministic_reruns_of_one_commit_do_not_replicate(repo, settings, make_trial):
    # same seed AND same commit => one replication unit => never enough to promote
    _persist_instability_group(repo, settings, make_trial, seeds=(42, 42, 42),
                               commits=["same", "same", "same"])
    assert promote_replications(repo, settings) == []


# --- multi-context C3/C4 accumulation (advance_promotions) -----------------------
def _ingest(settings, repo, commit, post, *, components=("optimizer",), config_hash=None, status="discard"):
    from failuretrace import record_rejected_trial

    return record_rejected_trial(
        {"git_commit": commit, "status": status, "baseline_metric": 1.0,
         "changed_components": list(components), "config_hash": config_hash,
         "hyperparameters": {"MATRIX_LR": 0.08}, "changed_hyperparameters": {"MATRIX_LR": 0.08}},
        {"post_change_metric": post}, "d",
        {"telemetry": {"gradient_norm_mean": 1.0, "gradient_norm_std": 3.0, "val_metric": post}, "finished": True},
        settings=settings, repository=repo,
    )


def test_advance_promotions_walks_full_ladder_across_contexts(make_env):
    from failuretrace import advance_promotions, link_counterfactual_trial

    settings, repo = make_env(ollama_enabled=False)
    # two same-family instability trials on distinct commits -> C1 -> C2 (auto-planned)
    _ingest(settings, repo, "c1", 1.15)
    _ingest(settings, repo, "c2", 1.15)
    rep = advance_promotions(repo, settings)["replication"]
    assert len(rep) == 1
    hyp_id = rep[0].hypothesis_id
    assert repo.effective_causal_level(hyp_id) == CausalSupportLevel.C2_replicated_effect

    # a counterfactual validation trial (LR reduced -> improvement) linked to the hypothesis
    cf1 = _ingest(settings, repo, "cf1", 0.9, config_hash="ctxA", status="completed")
    link_counterfactual_trial(repo, settings, hypothesis_id=hyp_id, counterfactual_trial_id=cf1.trial_id)
    step = advance_promotions(repo, settings)
    assert len(step["counterfactual"]) == 1 and not step["c4"]  # one context is not yet C4
    assert repo.effective_causal_level(hyp_id) == CausalSupportLevel.C3_counterfactual_supported

    # a second confirmation from a DISTINCT context -> C3 -> C4
    cf2 = _ingest(settings, repo, "cf2", 0.9, components=("data",), config_hash="ctxB", status="completed")
    link_counterfactual_trial(repo, settings, hypothesis_id=hyp_id, counterfactual_trial_id=cf2.trial_id)
    assert len(advance_promotions(repo, settings)["c4"]) == 1
    assert repo.effective_causal_level(hyp_id) == CausalSupportLevel.C4_robust_rule
    # idempotent: nothing more to promote
    assert not any(advance_promotions(repo, settings).values())


def test_c4_needs_two_distinct_contexts_not_just_count(make_env):
    from failuretrace import advance_promotions, link_counterfactual_trial

    settings, repo = make_env(ollama_enabled=False)
    _ingest(settings, repo, "c1", 1.15)
    _ingest(settings, repo, "c2", 1.15)
    hyp_id = advance_promotions(repo, settings)["replication"][0].hypothesis_id

    # two confirmations, but both from the SAME context -> reaches C3, never C4
    for commit in ("cf1", "cf1b"):
        cf = _ingest(settings, repo, commit, 0.9, config_hash="ctxA", status="completed")
        link_counterfactual_trial(repo, settings, hypothesis_id=hyp_id, counterfactual_trial_id=cf.trial_id)
    advance_promotions(repo, settings)  # -> C3
    assert repo.effective_causal_level(hyp_id) == CausalSupportLevel.C3_counterfactual_supported
    assert not advance_promotions(repo, settings)["c4"]  # 2 confirmations, 1 context -> no C4
    assert repo.effective_causal_level(hyp_id) == CausalSupportLevel.C3_counterfactual_supported


# --- promotion persists and raises the effective level (not the record) ----------
def test_replication_promotion_persists_and_raises_effective_level(repo, settings, make_trial):
    _, hyp = _persist_c1(repo, settings, make_trial)
    _promote_c2(repo, settings, make_trial, hyp)

    # the hypothesis record is unchanged; only the *effective* level moves.
    assert repo.get_hypothesis(hyp.hypothesis_id).causal_support_level == (
        CausalSupportLevel.C1_plausible_hypothesis
    )
    assert repo.effective_causal_level(hyp.hypothesis_id) == CausalSupportLevel.C2_replicated_effect
