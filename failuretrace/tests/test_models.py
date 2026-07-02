"""Model-validator tests (Phase 1): epistemic guardrails + derivations + improvement()."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from failuretrace import (
    CausalSupportLevel,
    FailureCategory,
    MetricDirection,
    PromotionRecord,
    improvement,
)
from failuretrace.core.ids import new_promotion_id


# --- confidence / quality bounds -------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("hypothesis_confidence", 1.5),
    ("hypothesis_confidence", -0.01),
    ("evidence_quality", 2.0),
    ("evidence_quality", -1.0),
])
def test_confidence_and_quality_bounded_0_1(make_hypothesis, field, value):
    with pytest.raises(ValidationError):
        make_hypothesis(**{field: value})


# --- single trial => C0/C1 only --------------------------------------------------
@pytest.mark.parametrize("level", [
    CausalSupportLevel.C2_replicated_effect,
    CausalSupportLevel.C3_counterfactual_supported,
    CausalSupportLevel.C4_robust_rule,
])
def test_fresh_hypothesis_cannot_assert_c2_plus(make_hypothesis, level):
    with pytest.raises(ValidationError):
        make_hypothesis(causal_support_level=level)


def test_c0_and_c1_are_allowed_at_creation(make_hypothesis):
    for level in (CausalSupportLevel.C0_observation, CausalSupportLevel.C1_plausible_hypothesis):
        h = make_hypothesis(causal_support_level=level)
        assert h.causal_support_level == level


# --- alternative explanations ----------------------------------------------------
def test_alt_explanations_required_for_nondeterministic(make_hypothesis):
    with pytest.raises(ValidationError):
        make_hypothesis(category=FailureCategory.likely_instability, alternative_explanations=[])


@pytest.mark.parametrize("category", [
    FailureCategory.divergence,
    FailureCategory.resource_pressure,
    FailureCategory.runtime_failure,
])
def test_alt_explanations_optional_for_deterministic(make_hypothesis, category):
    h = make_hypothesis(category=category, alternative_explanations=[])
    assert h.category == category


# --- hard constraint model-level guardrails -------------------------------------
def test_inconclusive_can_never_be_hard_constraint(make_hypothesis):
    with pytest.raises(ValidationError):
        make_hypothesis(
            category=FailureCategory.inconclusive,
            alternative_explanations=["noise"],
            should_apply_hard_constraint=True,
        )


def test_single_noisy_regression_cannot_be_hard_constraint(make_hypothesis):
    # likely_instability is not objectively deterministic => hard flag rejected on a record
    with pytest.raises(ValidationError):
        make_hypothesis(
            category=FailureCategory.likely_instability,
            should_apply_hard_constraint=True,
        )


def test_hard_constraint_flag_allowed_for_deterministic_category(make_hypothesis):
    # The model permits *setting* the flag for eligible categories; the repository still
    # gates on repeated/objective/C2 justification at write time (see test_stores).
    h = make_hypothesis(
        category=FailureCategory.resource_pressure,
        alternative_explanations=[],
        should_apply_hard_constraint=True,
    )
    assert h.should_apply_hard_constraint is True


def test_no_causal_confidence_field_exists(make_hypothesis):
    h = make_hypothesis()
    assert not hasattr(h, "causal_confidence")


# --- promotion monotonicity ------------------------------------------------------
def test_promotion_must_increase_level():
    with pytest.raises(ValidationError):
        PromotionRecord(
            promotion_id=new_promotion_id(),
            hypothesis_id="h1",
            from_level=CausalSupportLevel.C2_replicated_effect,
            to_level=CausalSupportLevel.C1_plausible_hypothesis,
            rationale="invalid downgrade",
            settings_hash="x",
        )


def test_promotion_target_must_be_c2_plus():
    with pytest.raises(ValidationError):
        PromotionRecord(
            promotion_id=new_promotion_id(),
            hypothesis_id="h1",
            from_level=CausalSupportLevel.C0_observation,
            to_level=CausalSupportLevel.C1_plausible_hypothesis,
            rationale="promotions only assert C2+",
            settings_hash="x",
        )


# --- direction-aware improvement() ----------------------------------------------
def test_improvement_minimize():
    assert improvement(1.0, 0.9, MetricDirection.minimize) == pytest.approx(0.1)   # better
    assert improvement(1.0, 1.1, MetricDirection.minimize) == pytest.approx(-0.1)  # worse


def test_improvement_maximize():
    assert improvement(0.9, 1.0, MetricDirection.maximize) == pytest.approx(0.1)   # better
    assert improvement(1.0, 0.9, MetricDirection.maximize) == pytest.approx(-0.1)  # worse


def test_identical_numbers_classify_oppositely_by_direction():
    # AC10 in miniature: same baseline/post, opposite sign under opposite directions.
    baseline, post = 1.0, 0.9
    assert improvement(baseline, post, MetricDirection.minimize) > 0
    assert improvement(baseline, post, MetricDirection.maximize) < 0


# --- TrialRecord derivations + immutability -------------------------------------
def test_metric_delta_and_peak_vram_derived(make_trial):
    t = make_trial(metric_delta=None, peak_vram_gb=None,
                   baseline_metric=1.0, post_change_metric=1.05,
                   telemetry={"peak_vram_gb": 44.0})
    assert t.metric_delta == pytest.approx(0.05)   # raw post - baseline
    assert t.peak_vram_gb == pytest.approx(44.0)   # copied from telemetry


def test_trial_record_is_immutable(make_trial):
    t = make_trial()
    with pytest.raises(ValidationError):
        t.post_change_metric = 2.0
