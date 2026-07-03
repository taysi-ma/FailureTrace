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

PROGRESS_LOG = (
    "step 00001 (0.1%) | loss: 4.000000 | lrm: 0.10 | dt: 310ms | "
    "tok/sec: 1,500,000 | mfu: 35.0% | epoch: 0 | remaining: 299s\r"
    "step 00002 (0.2%) | loss: 3.500000 | lrm: 0.20 | dt: 300ms | "
    "tok/sec: 1,600,000 | mfu: 39.0% | epoch: 0 | remaining: 298s\r"
    "step 00003 (0.3%) | loss: 3.800000 | lrm: 0.30 | dt: 290ms | "
    "tok/sec: 1,700,000 | mfu: 41.0% | epoch: 0 | remaining: 297s\n"
    "---\n"
    "val_bpb:          1.010000\n"
    "training_seconds: 300.0\n"
    "total_seconds:    320.0\n"
    "peak_vram_mb:     40960.0\n"
    "mfu_percent:      40.00\n"
    "total_tokens_M:   500.0\n"
    "num_steps:        3\n"
    "num_params_M:     51.5\n"
    "depth:            9\n"
)


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
    assert tel.mfu_percent == 39.80
    assert tel.total_seconds == 325.9
    assert tel.total_tokens_m == 499.6
    assert tel.num_steps == 953
    assert tel.num_params_m == 50.3
    assert tel.depth == 8
    assert tel.train_loss_start == 0.997900
    assert tel.train_loss_end == 0.997900
    assert tel.learning_rate_history == [0.0]
    assert tel.step_throughput_mean == 1_600_000
    assert tel.step_mfu_mean == 39.8


def test_parse_run_log_progress_carriage_returns():
    result = parse_run_log(PROGRESS_LOG)
    tel = result.telemetry
    assert tel.train_loss_start == 4.0
    assert tel.train_loss_end == 3.8
    assert tel.loss_spike_count == 1
    assert tel.learning_rate_history == [0.1, 0.2, 0.3]
    assert tel.step_throughput_min == 1_500_000
    assert tel.step_throughput_mean == 1_600_000
    assert tel.step_throughput_max == 1_700_000
    assert tel.step_mfu_min == 35.0
    assert tel.step_mfu_mean == 115.0 / 3.0
    assert tel.step_mfu_max == 41.0
    assert tel.num_steps == 3
    assert tel.num_params_m == 51.5
    assert tel.depth == 9


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
