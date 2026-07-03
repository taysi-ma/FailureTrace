"""Phase 7: controlled effect-size estimation.

Walks a hypothesis up to C3 (replication -> counterfactual promotion, with real persisted
counterfactual links) and checks the deterministic effect summary: magnitude, closed-form
interval, direction-awareness (minimize AND maximize), determinism, idempotency, and
physical immutability.
"""

from __future__ import annotations

import sqlite3

import pytest

from failuretrace import build_fallback, classify
from failuretrace.core.enums import CausalSupportLevel, MetricDirection
from failuretrace.estimation.effect import estimate_effect, estimate_effects
from failuretrace.planner.counterfactual import plan_counterfactual
from failuretrace.planner.replication import (
    ReplicationEvidence,
    evaluate_replication,
    link_counterfactual_trial,
    promote_counterfactuals,
)
from failuretrace.store.sqlite_store import connect
from failuretrace.tests.fixtures.scenarios import instability

_MIN = MetricDirection.minimize
_MAX = MetricDirection.maximize
_C3 = CausalSupportLevel.C3_counterfactual_supported


def _regress_post(direction):
    """A post-change metric that is a *regression* under ``direction`` (baseline 1.0)."""
    return 1.15 if direction == _MIN else 0.85


def _improve_post(direction, amount):
    """A post that improves by ``amount`` under ``direction`` (baseline 1.0)."""
    return 1.0 - amount if direction == _MIN else 1.0 + amount


def _src_ctx(direction):
    if direction == _MIN:
        return instability()
    return instability(metric_direction=_MAX, baseline_metric=1.0, post_change_metric=_regress_post(_MAX))


def _c1(repo, settings, make_trial, direction=_MIN, *, seed=42):
    trial = make_trial(
        trial_id="src", seed=seed, changed_components=["optimizer"],
        hyperparameters={"MATRIX_LR": 0.08}, metric_direction=direction,
        baseline_metric=1.0, post_change_metric=_regress_post(direction),
    )
    repo.save_trial(trial)
    ctx = _src_ctx(direction)
    hyp = build_fallback(classify(ctx, settings), ctx, trial_id=trial.trial_id, settings=settings)
    repo.save_hypothesis(hyp)
    return trial, hyp


def _to_c2(repo, settings, make_trial, hyp, direction=_MIN):
    evidence = []
    for s in (1, 2):
        t = make_trial(
            trial_id=f"rep{s}", seed=s, changed_components=["optimizer"],
            hyperparameters={"MATRIX_LR": 0.08}, metric_direction=direction,
            baseline_metric=1.0, post_change_metric=_regress_post(direction),
        )
        repo.save_trial(t)
        evidence.append(ReplicationEvidence(trial_id=t.trial_id, seed=s))
    repo.save_promotion(evaluate_replication(
        hyp.hypothesis_id, evidence, settings=settings, repository=repo, replication_group_id="g",
    ))


def _to_c3(repo, settings, make_trial, hyp, improvements, direction=_MIN):
    """``improvements``: list of (trial_id, amount) — each a counterfactual trial that
    improves by ``amount`` under ``direction`` and is linked to the hypothesis."""
    repo.save_plan(plan_counterfactual(hyp, settings=settings))
    for tid, amount in improvements:
        t = make_trial(
            trial_id=tid, changed_components=["optimizer"], hyperparameters={"MATRIX_LR": 0.04},
            metric_direction=direction, baseline_metric=1.0, post_change_metric=_improve_post(direction, amount),
        )
        repo.save_trial(t)
        link_counterfactual_trial(repo, settings, hypothesis_id=hyp.hypothesis_id, counterfactual_trial_id=t.trial_id)
    promote_counterfactuals(repo, settings)


def _full_c3(repo, settings, make_trial, improvements=(("cf1", 0.1), ("cf2", 0.2)), direction=_MIN):
    _, hyp = _c1(repo, settings, make_trial, direction)
    _to_c2(repo, settings, make_trial, hyp, direction)
    _to_c3(repo, settings, make_trial, hyp, list(improvements), direction)
    assert repo.effective_causal_level(hyp.hypothesis_id) == _C3
    return hyp


# --- magnitude + interval (minimize) --------------------------------------------
def test_effect_estimate_basic(repo, settings, make_trial):
    hyp = _full_c3(repo, settings, make_trial)
    est = estimate_effect(hyp.hypothesis_id, settings=settings, repository=repo)

    assert est is not None
    assert est.n_counterfactuals == 2
    assert est.absolute_effect == pytest.approx(0.15)     # mean of [0.1, 0.2], >0 = better
    assert est.relative_effect == pytest.approx(0.15)     # baseline 1.0
    assert est.range_low == pytest.approx(0.1)
    assert est.range_high == pytest.approx(0.2)
    assert est.consistency == 1.0                          # both deltas positive
    assert est.dispersion is not None and est.standardized_effect is not None
    assert est.ci_low is not None and est.ci_low < est.absolute_effect < est.ci_high


# --- direction-awareness: positive effect = "better" under maximize too ----------
def test_effect_estimate_respects_maximize(make_env, make_trial):
    settings, repo = make_env(metric={"name": "reward", "direction": "maximize"})
    hyp = _full_c3(repo, settings, make_trial, direction=_MAX)
    est = estimate_effect(hyp.hypothesis_id, settings=settings, repository=repo)

    assert est is not None
    assert est.metric_direction == _MAX
    assert est.absolute_effect == pytest.approx(0.15)     # post went UP under maximize, still +0.15
    assert est.consistency == 1.0


# --- interval only with enough replicates ---------------------------------------
def test_single_counterfactual_has_no_interval(repo, settings, make_trial):
    hyp = _full_c3(repo, settings, make_trial, improvements=(("cf1", 0.1),))
    est = estimate_effect(hyp.hypothesis_id, settings=settings, repository=repo)

    assert est.n_counterfactuals == 1
    assert est.absolute_effect == pytest.approx(0.1)
    assert est.ci_low is None and est.ci_high is None
    assert est.dispersion is None and est.standardized_effect is None
    assert est.range_low == pytest.approx(0.1) and est.range_high == pytest.approx(0.1)


# --- preconditions --------------------------------------------------------------
def test_below_c3_returns_none(repo, settings, make_trial):
    _, hyp = _c1(repo, settings, make_trial)
    _to_c2(repo, settings, make_trial, hyp)  # only C2
    assert estimate_effect(hyp.hypothesis_id, settings=settings, repository=repo) is None


def test_disabled_returns_none(make_env, make_trial):
    settings, repo = make_env(estimation={"enabled": False})
    hyp = _full_c3(repo, settings, make_trial)
    assert estimate_effect(hyp.hypothesis_id, settings=settings, repository=repo) is None
    assert estimate_effects(repo, settings) == []


# --- determinism: same records => identical numbers -----------------------------
def test_estimate_is_deterministic(repo, settings, make_trial):
    hyp = _full_c3(repo, settings, make_trial, improvements=(("cf1", 0.1), ("cf2", 0.25)))
    numeric = ("n_counterfactuals", "absolute_effect", "relative_effect", "standardized_effect",
               "dispersion", "ci_low", "ci_high", "range_low", "range_high", "consistency")
    a = estimate_effect(hyp.hypothesis_id, settings=settings, repository=repo)
    b = estimate_effect(hyp.hypothesis_id, settings=settings, repository=repo)
    assert {k: getattr(a, k) for k in numeric} == {k: getattr(b, k) for k in numeric}


# --- driver persists once, then is idempotent -----------------------------------
def test_estimate_effects_driver_is_idempotent(repo, settings, make_trial):
    hyp = _full_c3(repo, settings, make_trial)
    first = estimate_effects(repo, settings)
    assert len(first) == 1
    assert estimate_effects(repo, settings) == []  # same evidence => nothing new
    assert len(repo.list_effect_estimates_for_hypothesis(hyp.hypothesis_id)) == 1
    assert repo.latest_effect_estimate(hyp.hypothesis_id).absolute_effect == pytest.approx(0.15)


# --- the estimate feeds retrieval (additive, explained) -------------------------
def test_effect_boosts_and_explains_retrieval(repo, settings, make_trial):
    from failuretrace import InterventionContext, retrieve_relevant_failures
    from failuretrace.core.enums import FailureCategory

    hyp = _full_c3(repo, settings, make_trial)
    estimate_effects(repo, settings)
    ic = InterventionContext(category=FailureCategory.likely_instability, changed_components=["optimizer"])
    results = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    match = next(r for r in results if r.hypothesis.hypothesis_id == hyp.hypothesis_id)
    assert any("controlled effect" in line for line in match.score_explanation)


# --- physical immutability ------------------------------------------------------
def test_effect_estimate_is_immutable(repo, settings, make_trial):
    hyp = _full_c3(repo, settings, make_trial)
    estimate_effects(repo, settings)
    conn = connect(repo.sqlite.db_path)
    try:
        for op in ("UPDATE effect_estimates SET absolute_effect=0", "DELETE FROM effect_estimates"):
            with pytest.raises(sqlite3.Error):
                with conn:
                    conn.execute(op)
    finally:
        conn.close()
    assert repo.latest_effect_estimate(hyp.hypothesis_id).absolute_effect == pytest.approx(0.15)
