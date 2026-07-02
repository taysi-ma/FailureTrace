"""Stable, unique identifier generation for FailureTrace records.

IDs are opaque and unique per record. Record *identity* (dedup of a re-ingested
trial) is enforced by the stores (write-once), not by ID determinism.
"""

from __future__ import annotations

from uuid import uuid4


def _new(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def new_trial_id() -> str:
    return _new("trial")


def new_hypothesis_id() -> str:
    return _new("hyp")


def new_promotion_id() -> str:
    return _new("promo")


def new_plan_id() -> str:
    return _new("plan")


def new_link_id() -> str:
    return _new("link")


def new_replication_group_id() -> str:
    return _new("repl")
