"""Persistence layer: append-only SQLite + write-once JSON, behind a repository.

The repository is the only write path; it enforces immutability, write-once semantics,
and the hard-constraint write-time gate.
"""

from .errors import DuplicateRecordError, HardConstraintViolation, StoreError

__all__ = ["StoreError", "DuplicateRecordError", "HardConstraintViolation"]
