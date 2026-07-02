"""Telemetry tests: optional schema, CV derivation, collector, run.log adapter."""

from __future__ import annotations

from failuretrace.telemetry import TelemetryRecord, normalize, parse_run_log, telemetry_from_run_log

SUCCESS_LOG = """\
step 00953 (100.0%) | loss: 0.997900 | lrm: 0.00 | dt: 300ms | tok/sec: 1,600,000 | mfu: 39.8% | epoch: 0 | remaining: 0s
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45056.0
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
"""

OOM_LOG = """\
step 00007 (0.7%) | loss: 3.21 | ...
Traceback (most recent call last):
  File "train.py", line 552, in <module>
    loss.backward()
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
"""

FAIL_LOG = """\
step 00010 (1.0%) | loss: nan | ...
FAIL
"""


def test_all_fields_optional():
    rec = TelemetryRecord()
    assert rec.val_metric is None
    assert rec.present_fields() == set()


def test_gradient_norm_cv_derived():
    rec = TelemetryRecord(gradient_norm_mean=2.0, gradient_norm_std=4.0)
    assert rec.gradient_norm_cv == 2.0  # std / mean
    # explicit cv is not overwritten
    rec2 = TelemetryRecord(gradient_norm_mean=2.0, gradient_norm_std=4.0, gradient_norm_cv=9.9)
    assert rec2.gradient_norm_cv == 9.9


def test_normalize_is_partial_and_ignores_unknown_keys():
    rec = normalize({"val_metric": 1.0, "peak_vram_gb": 44.0, "not_a_field": 123})
    assert isinstance(rec, TelemetryRecord)
    assert rec.val_metric == 1.0
    assert rec.peak_vram_gb == 44.0
    assert normalize({}).runtime_seconds is None


def test_parse_run_log_success_summary():
    result = parse_run_log(SUCCESS_LOG)
    assert result.finished is True
    assert result.exception_type is None
    tel = result.telemetry
    assert tel.val_metric == 0.997900
    assert tel.runtime_seconds == 300.1
    assert tel.peak_vram_gb == 45056.0 / 1024  # == 44.0
    assert tel.throughput == 499.6 * 1e6 / 300.1
    assert tel.nan_detected is None


def test_parse_run_log_oom_traceback():
    result = parse_run_log(OOM_LOG)
    assert result.finished is False
    assert result.exception_type == "torch.cuda.OutOfMemoryError"
    assert "out of memory" in (result.exception_message or "").lower()


def test_parse_run_log_fail_marker_maps_to_nan():
    result = parse_run_log(FAIL_LOG)
    assert result.telemetry.nan_detected is True
    assert result.finished is False


def test_telemetry_from_run_log_helper():
    tel = telemetry_from_run_log(SUCCESS_LOG)
    assert isinstance(tel, TelemetryRecord)
    assert tel.val_metric == 0.997900
