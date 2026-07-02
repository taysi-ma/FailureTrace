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
from failuretrace.store.sqlite_store import connect


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
    assert versions == [1]  # each step applied exactly once


# --- write-once / immutability --------------------------------------------------
def test_trial_write_once(repo, make_trial):
    trial = make_trial()
    repo.save_trial(trial)
    with pytest.raises(DuplicateRecordError):
        repo.save_trial(trial)


def test_soft_hypothesis_saves_without_justification(repo, make_hypothesis):
    hyp = make_hypothesis(should_apply_soft_penalty=True)
    repo.save_hypothesis(hyp)
    stored = repo.get_hypothesis(hyp.hypothesis_id)
    assert stored is not None
    assert stored.should_apply_soft_penalty is True
    assert stored == hyp


# --- effective causal level via append-only promotion ---------------------------
def test_effective_causal_level_via_promotion(repo, make_hypothesis, settings):
    hyp = make_hypothesis()  # C1
    repo.save_hypothesis(hyp)
    assert repo.effective_causal_level(hyp.hypothesis_id) == (
        CausalSupportLevel.C1_plausible_hypothesis
    )

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


# --- hard-constraint write-time gate --------------------------------------------
def test_hard_constraint_refused_without_justification(repo, make_hypothesis):
    hyp = make_hypothesis(
        category=FailureCategory.resource_pressure,
        alternative_explanations=[],
        should_apply_hard_constraint=True,
    )
    with pytest.raises(HardConstraintViolation):
        repo.save_hypothesis(hyp)  # no repeated / objective limit / C2


def test_hard_constraint_allowed_when_objective_limit_exceeded(repo, make_hypothesis):
    hyp = make_hypothesis(
        category=FailureCategory.resource_pressure,
        alternative_explanations=[],
        should_apply_hard_constraint=True,
    )
    repo.save_hypothesis(hyp, telemetry={"peak_vram_gb": 80.0}, resource_limit_gb=79.0)
    assert repo.get_hypothesis(hyp.hypothesis_id) is not None


def test_hard_constraint_allowed_when_deterministic_and_repeated(repo, make_hypothesis):
    hyp = make_hypothesis(
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
