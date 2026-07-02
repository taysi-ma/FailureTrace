"""Store tests (Phase 1 gate): T5 round-trip, T16 idempotent init, write-once,
append-only promotions, hard-constraint gate, and settings_hash reproducibility."""

from __future__ import annotations

import pytest

from failuretrace import (
    CausalSupportLevel,
    DuplicateRecordError,
    FailureCategory,
    HardConstraintViolation,
    PromotionRecord,
    initialize_database,
    load_settings,
)
from failuretrace.core.ids import new_promotion_id
from failuretrace.store.errors import PromotionViolation
from failuretrace.store.sqlite_store import connect


def _c1(hyp_id, settings, **over):
    base = dict(
        promotion_id=new_promotion_id(),
        hypothesis_id=hyp_id,
        from_level=CausalSupportLevel.C1_plausible_hypothesis,
        to_level=CausalSupportLevel.C2_replicated_effect,
        supporting_trial_ids=["t1", "t2"],
        rationale="two matched-seed replications",
        settings_hash=settings.settings_hash(),
    )
    base.update(over)
    return PromotionRecord(**base)


def _schema_snapshot(db_path):
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
    finally:
        conn.close()
    return [tuple(r) for r in rows]


# --- T5: store round-trip through SQLite AND JSON --------------------------------
def test_t5_store_round_trip_sqlite_and_json(repo, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    assert repo.get_trial(trial.trial_id) == trial          # SQLite
    assert repo.json.read_trial(trial.trial_id) == trial     # raw JSON
    assert repo.json.exists(trial.trial_id)


# --- T16: initialize_database is idempotent -------------------------------------
def test_t16_initialize_database_idempotent(settings):
    db1 = initialize_database(settings)
    snapshot1 = _schema_snapshot(db1)
    db2 = initialize_database(settings)  # second call must not error
    snapshot2 = _schema_snapshot(db2)

    assert db1 == db2
    assert snapshot1 == snapshot2  # identical schema

    conn = connect(db1)
    try:
        versions = [r["version"] for r in conn.execute(
            "SELECT version FROM schema_version ORDER BY version")]
    finally:
        conn.close()
    assert versions == [1, 2, 3]  # each step applied exactly once


# --- write-once / immutability --------------------------------------------------
def test_trial_write_once(repo, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    with pytest.raises(DuplicateRecordError):
        repo.save_trial(trial)


def test_soft_hypothesis_saves_without_justification(repo, make_hypothesis, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    hyp = make_hypothesis(trial_id=trial.trial_id, should_apply_soft_penalty=True)
    repo.save_hypothesis(hyp)
    stored = repo.get_hypothesis(hyp.hypothesis_id)
    assert stored is not None
    assert stored.should_apply_soft_penalty is True
    assert stored == hyp


def test_hypothesis_referencing_unknown_trial_is_refused(repo, make_hypothesis):
    from failuretrace import StoreError

    hyp = make_hypothesis(trial_id="trial_does_not_exist")
    with pytest.raises(StoreError):  # ReferentialIntegrityError
        repo.save_hypothesis(hyp)


# --- effective causal level via append-only promotion ---------------------------
def test_effective_causal_level_via_promotion(repo, make_hypothesis, settings, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    hyp = make_hypothesis(trial_id=trial.trial_id)  # C1
    repo.save_hypothesis(hyp)
    assert repo.effective_causal_level(hyp.hypothesis_id) == (
        CausalSupportLevel.C1_plausible_hypothesis
    )

    # supporting trials must be real (write-path gate + FK)
    t1 = make_trial(trial_id="t1", seed=1)
    t2 = make_trial(trial_id="t2", seed=2)
    repo.save_trial(t1)
    repo.save_trial(t2)
    repo.save_promotion(
        PromotionRecord(
            promotion_id=new_promotion_id(),
            hypothesis_id=hyp.hypothesis_id,
            from_level=CausalSupportLevel.C1_plausible_hypothesis,
            to_level=CausalSupportLevel.C2_replicated_effect,
            supporting_trial_ids=["t1", "t2"],
            rationale="two matched-seed replications",
            settings_hash=settings.settings_hash(),
        )
    )
    # hypothesis record itself is unchanged; only the *effective* level moves.
    assert repo.get_hypothesis(hyp.hypothesis_id).causal_support_level == (
        CausalSupportLevel.C1_plausible_hypothesis
    )
    assert repo.effective_causal_level(hyp.hypothesis_id) == (
        CausalSupportLevel.C2_replicated_effect
    )


# --- promotion write-time evidence gate (audit probes 1 & 3) --------------------
def test_promotion_for_unknown_hypothesis_is_refused(repo, settings):
    with pytest.raises(PromotionViolation):
        repo.save_promotion(_c1("hyp_does_not_exist", settings))


def test_promotion_with_fabricated_supporting_trials_is_refused(repo, make_hypothesis, make_trial, settings):
    trial = make_trial()
    repo.save_trial(trial)
    hyp = make_hypothesis(trial_id=trial.trial_id)
    repo.save_hypothesis(hyp)
    # supporting trials "t1"/"t2" were never persisted -> the gate refuses the promotion
    with pytest.raises(PromotionViolation):
        repo.save_promotion(_c1(hyp.hypothesis_id, settings))


def test_promotion_ladder_cannot_be_skipped(repo, make_hypothesis, make_trial, settings):
    trial = make_trial()
    repo.save_trial(trial)
    hyp = make_hypothesis(trial_id=trial.trial_id)  # effective level C1
    repo.save_hypothesis(hyp)
    t1, t2 = make_trial(trial_id="t1", seed=1), make_trial(trial_id="t2", seed=2)
    repo.save_trial(t1)
    repo.save_trial(t2)
    # a C2->C3 promotion while the hypothesis is still effectively C1 must be refused
    with pytest.raises(PromotionViolation):
        repo.save_promotion(_c1(
            hyp.hypothesis_id, settings,
            from_level=CausalSupportLevel.C2_replicated_effect,
            to_level=CausalSupportLevel.C3_counterfactual_supported,
        ))


# --- hard-constraint write-time gate --------------------------------------------
def test_hard_constraint_refused_without_justification(repo, make_hypothesis, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    hyp = make_hypothesis(
        trial_id=trial.trial_id,
        category=FailureCategory.resource_pressure,
        alternative_explanations=[],
        should_apply_hard_constraint=True,
    )
    with pytest.raises(HardConstraintViolation):
        repo.save_hypothesis(hyp)  # no repeated / objective limit / C2


def test_hard_constraint_allowed_when_objective_limit_exceeded(repo, make_hypothesis, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    hyp = make_hypothesis(
        trial_id=trial.trial_id,
        category=FailureCategory.resource_pressure,
        alternative_explanations=[],
        should_apply_hard_constraint=True,
    )
    repo.save_hypothesis(hyp, telemetry={"peak_vram_gb": 80.0}, resource_limit_gb=79.0)
    assert repo.get_hypothesis(hyp.hypothesis_id) is not None


def test_hard_constraint_allowed_when_deterministic_and_repeated(repo, make_hypothesis, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    hyp = make_hypothesis(
        trial_id=trial.trial_id,
        category=FailureCategory.resource_pressure,
        alternative_explanations=[],
        should_apply_hard_constraint=True,
    )
    repo.save_hypothesis(hyp, repeated=True)
    assert repo.get_hypothesis(hyp.hypothesis_id) is not None


# --- settings_hash reproducibility ----------------------------------------------
def test_settings_hash_excludes_paths_includes_metric_direction(tmp_path):
    a = load_settings(
        overrides={"paths": {"data_dir": str(tmp_path / "a"), "reports_dir": str(tmp_path / "r")}},
        env={},
    )
    b = load_settings(
        overrides={"paths": {"data_dir": str(tmp_path / "b"), "reports_dir": str(tmp_path / "r2")}},
        env={},
    )
    assert a.settings_hash() == b.settings_hash()  # paths are environment-specific

    c = load_settings(
        overrides={
            "paths": {"data_dir": str(tmp_path / "a"), "reports_dir": str(tmp_path / "r")},
            "metric": {"name": "val_bpb", "direction": "maximize"},
        },
        env={},
    )
    assert a.settings_hash() != c.settings_hash()  # metric direction is semantic
