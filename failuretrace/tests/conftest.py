"""Shared fixtures and record factories for the FailureTrace test suite.

Factories are exposed as fixtures (callables) so later phases can reuse them. Settings
are always loaded with ``env={}`` and an explicit temp ``data_dir`` so tests are
isolated from any ambient ``FAILURETRACE_DATA_DIR``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from failuretrace import (
    CausalSupportLevel,
    CounterfactualPlanRef,
    FailureCategory,
    FailureHypothesis,
    HypothesisSource,
    Intervention,
    MetricDirection,
    Repository,
    TrialRecord,
    TrialStatus,
    initialize_database,
    load_settings,
)
from failuretrace.core.ids import new_hypothesis_id, new_trial_id

_FIXED_TS = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings(tmp_path):
    return load_settings(
        overrides={
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "reports_dir": str(tmp_path / "reports"),
            }
        },
        env={},
    )


@pytest.fixture
def repo(settings):
    initialize_database(settings)
    return Repository(settings)


@pytest.fixture
def make_trial():
    def _factory(**over) -> TrialRecord:
        base = dict(
            trial_id=new_trial_id(),
            timestamp=_FIXED_TS,
            git_commit="a1b2c3d",
            config_hash="cfg-hash-1",
            seed=42,
            status=TrialStatus.rejected,
            metric_name="val_bpb",
            metric_direction=MetricDirection.minimize,
            baseline_metric=1.0,
            post_change_metric=1.05,
            runtime_seconds=300.0,
            throughput=1_500_000.0,
            changed_files=["train.py"],
            changed_components=["optimizer"],
            hyperparameters={"MATRIX_LR": 0.04, "DEPTH": 8},
            telemetry={"peak_vram_gb": 44.0, "val_metric": 1.05},
        )
        base.update(over)
        return TrialRecord(**base)

    return _factory


@pytest.fixture
def make_hypothesis(settings):
    def _factory(**over) -> FailureHypothesis:
        base = dict(
            hypothesis_id=new_hypothesis_id(),
            trial_id="trial_example",
            source=HypothesisSource.rule_based,
            category=FailureCategory.likely_instability,
            observations=["gradient_norm_cv above threshold; val regressed"],
            evidence=["gradient_norm_cv=2.5"],
            hypotheses=["learning rate too high for this configuration"],
            alternative_explanations=["measurement noise", "data ordering"],
            missing_evidence=["gradient_norm history"],
            hypothesis_confidence=0.5,
            evidence_quality=0.5,
            suggested_intervention=Intervention(
                variable="optimizer.lr", action="decrease", target_value=0.02,
                rationale="halve the matrix LR to test instability",
            ),
            proposed_counterfactual_trial=CounterfactualPlanRef(
                summary="hold code + effective batch fixed; halve LR",
            ),
            causal_support_level=CausalSupportLevel.C1_plausible_hypothesis,
            settings_hash=settings.settings_hash(),
        )
        base.update(over)
        return FailureHypothesis(**base)

    return _factory
