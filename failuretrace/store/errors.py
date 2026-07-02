"""Store-layer exceptions (kept import-free to avoid cycles)."""

from __future__ import annotations


class StoreError(RuntimeError):
    """Base class for persistence errors."""


class DuplicateRecordError(StoreError):
    """Raised when an append-only / write-once record already exists."""


class HardConstraintViolation(StoreError):
    """Raised when a hard-constraint record fails its write-time justification gate."""
