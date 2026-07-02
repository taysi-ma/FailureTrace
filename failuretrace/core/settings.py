"""Settings: load ``config/defaults.yaml`` into a typed, hashable configuration object.

This module also hosts the single canonical, direction-aware comparison helper
:func:`improvement`. No comparison logic anywhere else in the package may assume
"larger is better" — it must call :func:`improvement`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .enums import MetricDirection

logger = logging.getLogger(__name__)

ENV_DATA_DIR = "FAILURETRACE_DATA_DIR"

# Config sections that define reproducible *result identity*. Deliberately EXCLUDES
# environment-specific paths and feature flags so that moving the data dir or toggling
# a flag does not change historical result hashes. Grows as later phases add sections.
_SEMANTIC_SECTIONS = ("metric", "thresholds", "confidence", "retrieval")


class Paths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path
    reports_dir: Path


class MetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    direction: MetricDirection


class Settings(BaseModel):
    """Effective FailureTrace configuration.

    ``extra="allow"`` lets later phases add config sections (thresholds, confidence,
    retrieval, ollama, ...) without a schema change here; they are hashed via
    :meth:`settings_hash` when semantically relevant.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    ollama_enabled: bool = False
    collect_telemetry: bool = True
    store_raw_json: bool = True
    counterfactual_planner_enabled: bool = True
    replication_gate_enabled: bool = True

    paths: Paths
    metric: MetricConfig

    # The raw merged config dict, retained for a stable semantic hash.
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)

    def settings_hash(self) -> str:
        """Stable 16-hex hash over the *semantic* configuration.

        Includes ``metric`` (name + direction) plus any ``thresholds``/``confidence``/
        ``retrieval`` sections present. Stored with every classification, hypothesis,
        plan, and promotion so results remain reproducible if ``defaults.yaml`` changes.
        """
        source = self._raw or self.model_dump(mode="json")
        semantic: dict[str, Any] = {k: source[k] for k in _SEMANTIC_SECTIONS if k in source}
        semantic.setdefault("metric", self.metric.model_dump(mode="json"))
        canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def section(self, name: str) -> dict[str, Any]:
        """Return a config section (e.g. ``thresholds``, ``confidence``) as a dict.

        Reads from the raw merged config so later-phase sections are available even
        though they are not modeled as explicit ``Settings`` fields.
        """
        source = self._raw or self.model_dump(mode="python")
        value = source.get(name, {})
        return dict(value) if isinstance(value, dict) else {}


def default_config_path() -> Path:
    """Absolute path to the packaged ``defaults.yaml``."""
    return Path(str(resources.files("failuretrace.config").joinpath("defaults.yaml")))


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} must be a YAML mapping, got {type(data).__name__}")
    return data


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(
    *,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> Settings:
    """Load settings from YAML, applying (in ascending precedence) file defaults,
    the ``FAILURETRACE_DATA_DIR`` environment variable, then explicit ``overrides``.
    """
    env = os.environ if env is None else env
    raw = _read_yaml(Path(config_path) if config_path else default_config_path())

    data_dir_env = env.get(ENV_DATA_DIR)
    if data_dir_env:
        raw.setdefault("paths", {})
        raw["paths"]["data_dir"] = data_dir_env

    if overrides:
        raw = _deep_merge(raw, overrides)

    settings = Settings(**raw)
    settings._raw = raw
    logger.debug("loaded settings (hash=%s, data_dir=%s)", settings.settings_hash(), settings.paths.data_dir)
    return settings


_cached: Settings | None = None


def get_settings(*, reload: bool = False) -> Settings:
    """Return a process-cached default Settings (loaded from the packaged YAML)."""
    global _cached
    if _cached is None or reload:
        _cached = load_settings()
    return _cached


def improvement(baseline: float, post: float, direction: MetricDirection) -> float:
    """Direction-aware improvement. Positive = improvement, negative = regression.

    - ``minimize``: ``baseline - post`` (a lower post-change metric is better).
    - ``maximize``: ``post - baseline`` (a higher post-change metric is better).

    This is the single canonical helper; all comparison logic (classifier, gate,
    retrieval, planner, reports) must route through it. Never assume larger is better.
    """
    if direction == MetricDirection.minimize:
        return baseline - post
    return post - baseline
