"""Store-layer exceptions (kept import-free to avoid cycles)."""

from __future__ import annotations


class StoreError(RuntimeError):
    """Base class for persistence errors."""


class DuplicateRecordError(StoreError):
    """Raised when an append-only / write-once record already exists."""


class HardConstraintViolation(StoreError):
    """Raised when a hard-constraint record fails its write-time justification gate."""


class ReferentialIntegrityError(StoreError):
    """Raised when a record references a parent row that does not exist.

    Enforced at the database level by foreign keys (schema v3) and, with a clearer
    message, at the application level by the repository write-path gates.
    """


class PromotionViolation(StoreError):
    """Raised when a ``PromotionRecord`` fails its write-time evidence gate.

    The repository refuses to persist a promotion whose stated ``from_level`` does not
    match the hypothesis's current effective level (ladder integrity), whose supporting
    trials do not exist, or which does not carry the minimum supporting evidence for its
    target level. This makes causal-support upgrades non-forgeable at the write path,
    independent of whatever the (also-hardened) evaluators computed.
    """
