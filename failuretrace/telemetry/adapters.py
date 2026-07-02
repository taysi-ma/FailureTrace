"""Adapters from trainer-specific artifacts to normalized telemetry.

Implements the autoresearch adapter from the Phase-0 report: autoresearch's ``train.py``
prints a summary block to stdout (captured in ``run.log``) and a ``FAIL`` marker on
NaN / loss-blow-up; crashes surface as Python tracebacks. There is no structured
sidecar, so we parse ``run.log`` text.

Mapping (autoresearch -> TelemetryRecord):
- ``val_bpb``           -> ``val_metric``
- ``training_seconds``  -> ``runtime_seconds``
- ``peak_vram_mb``/1024 -> ``peak_vram_gb``
- ``total_tokens_M`` * 1e6 / ``training_seconds`` -> ``throughput`` (tokens/sec)
- ``FAIL`` marker       -> ``nan_detected=True`` (autoresearch's guard fires on NaN or
                           train loss > 100; run.log does not distinguish the two)
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, ConfigDict

from .schema import TelemetryRecord

logger = logging.getLogger(__name__)

# Numeric keys autoresearch prints in its final summary block (train.py:621-630).
_SUMMARY_KEYS = frozenset(
    {
        "val_bpb",
        "training_seconds",
        "total_seconds",
        "peak_vram_mb",
        "mfu_percent",
        "total_tokens_M",
        "num_steps",
        "num_params_M",
        "depth",
    }
)

_SUMMARY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")
_FAIL_MARKER = re.compile(r"^FAIL\s*$", re.MULTILINE)
_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt)):[ \t]?(.*)$", re.MULTILINE)
_OOM_HINT = re.compile(r"out of memory|CUDA out of memory|OutOfMemory", re.IGNORECASE)


class RunLogParse(BaseModel):
    """Structured result of parsing a run log."""

    model_config = ConfigDict(extra="forbid")

    telemetry: TelemetryRecord
    exception_type: str | None = None
    exception_message: str | None = None
    finished: bool = False
    summary: dict[str, float] = {}


def _parse_summary(text: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    for line in text.splitlines():
        match = _SUMMARY_LINE.match(line)
        if match and match.group(1) in _SUMMARY_KEYS:
            summary[match.group(1)] = float(match.group(2))
    return summary


def _parse_exception(text: str) -> tuple[str | None, str | None]:
    matches = _EXC_LINE.findall(text)
    if matches:
        exc_type, message = matches[-1]  # last exception line = the raised one
        return exc_type, (message.strip() or None)
    if _OOM_HINT.search(text):
        return "OutOfMemoryError", "CUDA out of memory"
    return None, None


def telemetry_from_run_log(text: str) -> TelemetryRecord:
    """Convenience: parse a run log and return only its normalized telemetry."""
    return parse_run_log(text).telemetry


def parse_run_log(text: str) -> RunLogParse:
    """Parse an autoresearch ``run.log`` into normalized telemetry + crash info."""
    summary = _parse_summary(text)
    fail = bool(_FAIL_MARKER.search(text))
    exc_type, exc_message = _parse_exception(text)

    peak_vram_gb = summary["peak_vram_mb"] / 1024 if "peak_vram_mb" in summary else None
    training_seconds = summary.get("training_seconds")
    throughput = None
    if summary.get("total_tokens_M") is not None and training_seconds:
        throughput = summary["total_tokens_M"] * 1e6 / training_seconds

    telemetry = TelemetryRecord(
        val_metric=summary.get("val_bpb"),
        runtime_seconds=training_seconds,
        peak_vram_gb=peak_vram_gb,
        throughput=throughput,
        nan_detected=True if fail else None,
    )

    # "finished" == produced a val_bpb summary and did not raise.
    finished = "val_bpb" in summary and exc_type is None
    return RunLogParse(
        telemetry=telemetry,
        exception_type=exc_type,
        exception_message=exc_message,
        finished=finished,
        summary=summary,
    )
