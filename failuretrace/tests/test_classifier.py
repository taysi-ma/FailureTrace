"""Classifier tests including Gate-2 checks T1-T4, T14, T17."""

from __future__ import annotations

import sys

import pytest

from failuretrace.classifier import ClassificationContext, classify
from failuretrace.core.enums import FailureCategory, MetricDirection
from failuretrace.telemetry import TelemetryRecord
from failuretrace.tests.fixtures.scenarios import (
    SCENARIOS,
    divergence_nan,
    instability,
    missing_telemetry,
    oom_crash,
    over_regularization,
)


# --- every scenario classifies as intended -------------------------------------
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenarios_classify_as_expected(settings, name):
    factory, expected = SCENARIOS[name]
    result = classify(factory(), settings)
    assert result.category == expected


# --- T1: NaN/Inf -> divergence --------------------------------------------------
def test_t1_nan_inf_is_divergence(settings):
    assert classify(divergence_nan(), settings).category == FailureCategory.divergence
    inf_ctx = ClassificationContext(
        baseline_metric=1.0, post_change_metric=2.0,
        telemetry=TelemetryRecord(inf_detected=True), finished=False,
    )
    assert classify(inf_ctx, settings).category == FailureCategory.divergence


# --- T2: OOM -> resource_pressure ----------------------------------------------
def test_t2_oom_is_resource_pressure(settings):
    assert classify(oom_crash(), settings).category == FailureCategory.resource_pressure
    # also via the gpu_memory_ratio threshold branch
    ratio_ctx = ClassificationContext(
        baseline_metric=1.0, post_change_metric=1.2,
        telemetry=TelemetryRecord(gpu_memory_ratio=0.99),
    )
    assert classify(ratio_ctx, settings).category == FailureCategory.resource_pressure


# --- T3: high grad CV + regression -> likely_instability ------------------------
def test_t3_high_gradcv_regression_is_instability(settings):
    result = classify(instability(), settings)
    assert result.category == FailureCategory.likely_instability
    assert result.confidence == pytest.approx(0.7)  # strong_heuristic, full evidence


# --- T4: missing telemetry never crashes; degrades with reduced confidence -------
@pytest.mark.parametrize("ctx_factory", [
    lambda: missing_telemetry(),                            # |improvement| ~ 0 -> inconclusive
    lambda: missing_telemetry(post_change_metric=1.5),     # regression, no signal -> unknown
])
def test_t4_missing_telemetry_degrades_gracefully(settings, ctx_factory):
    result = classify(ctx_factory(), settings)
    assert result.category in {FailureCategory.inconclusive, FailureCategory.unknown}
    assert result.confidence <= 0.5  # reduced vs deterministic 0.95
    assert result.observations  # still explainable


# --- T14: identical numbers classify oppositely by direction --------------------
def test_t14_direction_flips_classification(settings):
    telemetry = TelemetryRecord(gradient_norm_mean=1.0, gradient_norm_std=3.0)  # cv = 3.0
    common = dict(baseline_metric=1.0, post_change_metric=1.1, telemetry=telemetry)
    minimize = ClassificationContext(metric_direction=MetricDirection.minimize, **common)
    maximize = ClassificationContext(metric_direction=MetricDirection.maximize, **common)
    assert classify(minimize, settings).category == FailureCategory.likely_instability
    assert classify(maximize, settings).category != FailureCategory.likely_instability


# --- T17: CPU-only; GPU metrics optional, no torch imported ----------------------
def test_t17_cpu_only(settings):
    assert classify(oom_crash(), settings).category == FailureCategory.resource_pressure
    assert classify(instability(), settings).category == FailureCategory.likely_instability
    assert "torch" not in sys.modules


# --- confidence rubric ----------------------------------------------------------
def test_deterministic_confidence_is_full(settings):
    assert classify(divergence_nan(), settings).confidence == 0.95


def test_completeness_caps_confidence(settings):
    full = classify(instability(), settings)  # cv + mean + std present
    sparse = classify(instability(telemetry=TelemetryRecord(gradient_norm_cv=3.0)), settings)
    assert full.category == sparse.category == FailureCategory.likely_instability
    assert sparse.confidence < full.confidence  # missing mean/std reduces confidence


def test_classification_is_explainable_and_stamped(settings):
    result = classify(instability(), settings)
    assert result.observations and result.triggered_rules
    assert result.settings_hash == settings.settings_hash()


def test_over_regularization_needs_train_regression_evidence(settings):
    # with train metrics showing regression, the rule fires (both train and val worsen)
    assert classify(over_regularization(), settings).category == (
        FailureCategory.possible_over_regularization
    )
    # without train metrics we cannot confirm "both worsen" -> must NOT claim over-regularization
    val_only = over_regularization(baseline_train_metric=None, post_train_metric=None)
    assert classify(val_only, settings).category != FailureCategory.possible_over_regularization


def test_alternative_categories_recorded(settings):
    # gpu ratio pressure + gradient instability both fire; highest priority wins,
    # the other appears as an alternative.
    ctx = ClassificationContext(
        baseline_metric=1.0, post_change_metric=1.2,
        telemetry=TelemetryRecord(gpu_memory_ratio=0.99, gradient_norm_mean=1.0, gradient_norm_std=3.0),
    )
    result = classify(ctx, settings)
    assert result.category == FailureCategory.resource_pressure
    assert FailureCategory.likely_instability in result.alternative_categories
