"""Build normalized telemetry records from partial inputs, gracefully.

Downstream code depends only on :class:`TelemetryRecord`, never on raw trainer logs
(those are handled by :mod:`failuretrace.telemetry.adapters`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from .schema import TelemetryRecord

logger = logging.getLogger(__name__)


def normalize(data: Mapping[str, Any] | None = None, **kwargs: Any) -> TelemetryRecord:
    """Construct a :class:`TelemetryRecord` from a mapping and/or keywords.

    Unknown keys are ignored (so a raw trainer dict can be passed directly); missing
    keys stay ``None``. Never raises on partial input.
    """
    merged: dict[str, Any] = dict(data or {})
    merged.update(kwargs)
    known = {k: v for k, v in merged.items() if k in TelemetryRecord.model_fields}
    dropped = set(merged) - set(known)
    if dropped:
        logger.debug("normalize() ignored unknown telemetry keys: %s", sorted(dropped))
    return TelemetryRecord(**known)
