"""The autoresearch adapter (Phase 0 §3.7).

Public API (spec §5.3), fed by the adapter with real artifacts:

    failuretrace.record_rejected_trial(experiment_context, metrics, diff, runtime_diagnostics)

`experiment_context`, `metrics`, and `runtime_diagnostics` are plain mappings (the real
repo has no fixed object to pass), documented below. `runtime_diagnostics` may also be a
``RunLogParse`` from :func:`failuretrace.telemetry.parse_run_log`.

Two feeding paths, both zero-touch to autoresearch:
- **Live hook (primary/complete)** — an optional, flag-guarded ``program.md`` section
  (:func:`render_program_md_hook`) tells the agent to call ``python -m failuretrace
  record`` at the moment of rejection, before ``git reset``/log-overwrite. Driven by
  :func:`record_from_run`.
- **Offline batch (best-effort/lossy)** — :func:`ingest_results_tsv` reads a preserved
  ``results.tsv`` + working tree for backfill.

When ``settings.enabled`` is false everything here is a guarded no-op: the hook renders
to ``None`` (so autoresearch never invokes FailureTrace and never imports its internals),
and :func:`record_rejected_trial` returns ``None`` without any DB/JSON write.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core.enums import TrialStatus
from ..core.ids import new_trial_id
from ..core.models import TrialRecord
from ..core.settings import Settings
from ..telemetry.adapters import RunLogParse

logger = logging.getLogger(__name__)

# Status strings autoresearch writes to results.tsv (program.md:77) -> TrialStatus.
_STATUS_MAP = {
    "keep": TrialStatus.promoted,
    "promoted": TrialStatus.promoted,
    "discard": TrialStatus.rejected,
    "rejected": TrialStatus.rejected,
    "completed": TrialStatus.completed,
    "invalid": TrialStatus.invalid,
    "inconclusive": TrialStatus.inconclusive,
}

_HOOK_TEMPLATE = """<!-- failuretrace:begin (present only when failuretrace.enabled) -->
### Optional: capture rejected/crashed trials as negative evidence (FailureTrace)

After step 7 (append the `results.tsv` row) and BEFORE any `git reset` or the next run,
if the status is `discard` or `crash`, record the trial:

```
python -m failuretrace record --commit <hash> --status <discard|crash> \\
    --run-log run.log --repo . --branch {branch} --description "<desc>"
```

This reads `run.log` + the `git diff` + the `results.tsv` row and stores a `TrialRecord`
plus a deterministic failure hypothesis. It NEVER edits `train.py`, changes the metric,
or alters the keep/reset decision.
<!-- failuretrace:end -->
"""


_CONSULT_TEMPLATE = """<!-- failuretrace:consult:begin (present only when failuretrace.enabled) -->
### Optional: consult prior failures before running (FailureTrace)

After editing `train.py` (step 2) and BEFORE `git commit` and the training run, check what
previous rejected trials say about this change:

```
python -m failuretrace brief --infer-from {repo}
```

This infers which tunables you changed by diffing the working tree against `HEAD`, then
prints the ranked prior failures for that change. How to read it:

- **Binding constraints** — backed by replicated (C2+) or repeated deterministic evidence.
  Treat these as forbidden regions; pick a different change.
- **Advisory** — a soft penalty. Prefer an alternative, but proceed if you have a reason.
- **Plausible hypotheses (C0/C1)** — single-trial observations, NOT causally validated.
  Context only. Do not treat them as established facts, and do not let them stop you
  from testing an idea.

This command only reads; it never edits `train.py`, changes the metric, or alters the
keep/reset decision.
<!-- failuretrace:consult:end -->
"""


def render_program_md_hook(settings: Settings, *, branch: str = "autoresearch/<tag>") -> str | None:
    """The flag-guarded ``program.md`` snippet, or ``None`` when disabled.

    Disabled ⇒ the section is absent ⇒ autoresearch never calls FailureTrace (AC13).
    """
    if not settings.enabled:
        return None
    return _HOOK_TEMPLATE.format(branch=branch)


def render_program_md_consult_hook(settings: Settings, *, repo: str = ".") -> str | None:
    """The read-path counterpart to :func:`render_program_md_hook`.

    The record hook closes the write half of the loop (a rejection becomes stored
    evidence); this one closes the read half (stored evidence reaches the agent before
    it commits to the next experiment). Same flag discipline: ``None`` when disabled, so
    autoresearch never learns FailureTrace exists (AC13).
    """
    if not settings.enabled:
        return None
    return _CONSULT_TEMPLATE.format(repo=repo)


# Word-boundary OOM detection (avoids false positives like "boom" containing "oom").
_OOM_RE = re.compile(r"\boom\b|out ?of ?memory", re.IGNORECASE)


def _is_oom(exception_type: str | None, message: str | None) -> bool:
    return bool(_OOM_RE.search(f"{exception_type or ''} {message or ''}"))


def _resolve_status(status: str | None, exception_type: str | None, message: str | None) -> TrialStatus:
    key = (status or "").strip().lower()
    if key in _STATUS_MAP:
        return _STATUS_MAP[key]
    if key == "crash" or exception_type:
        return TrialStatus.failed_oom if _is_oom(exception_type, message) else TrialStatus.failed_runtime
    # No explicit verdict and it finished cleanly ⇒ this API is for rejects: default rejected.
    return TrialStatus.rejected


def _summarize_diff(diff: str | None, *, max_chars: int = 400) -> str | None:
    if not diff:
        return None
    stripped = diff.strip()
    return stripped if len(stripped) <= max_chars else stripped[:max_chars] + " …"


def record_rejected_trial(
    experiment_context: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
    diff: str | None,
    runtime_diagnostics: Mapping[str, Any] | RunLogParse | None,
    *,
    settings: Settings,
    repository: Any = None,
) -> TrialRecord | None:
    """Persist a rejected/crashed trial + a deterministic failure hypothesis.

    Mapping contracts (all keys optional):
    - ``experiment_context``: ``git_commit``/``commit``, ``parent_trial_id``, ``seed``,
      ``status`` (keep|discard|crash|…), ``baseline_metric``, ``config_hash``,
      ``changed_files``, ``changed_components``, ``hyperparameters``,
      ``baseline_hyperparameters``, ``changed_hyperparameters``, invalid-comparison
      signals (``baseline_metric_name``/``post_metric_name``/``*_seed``/``*_config_hash``/
      ``requires_matched_seeds``), ``description``.
    - ``metrics``: ``post_change_metric`` (or the configured metric name, e.g. ``val_bpb``),
      ``throughput``, ``runtime_seconds``.
    - ``runtime_diagnostics``: a ``RunLogParse`` OR a mapping with ``telemetry`` (dict),
      ``exception_type``, ``exception_message``, ``finished``.

    Guarded no-op (returns ``None``) when ``settings.enabled`` is false.
    """
    if not settings.enabled:
        logger.info("failuretrace disabled; record_rejected_trial is a no-op")
        return None

    # Heavy imports kept out of the disabled path.
    from ..analyst import analyze
    from ..classifier import ClassificationContext, classify
    from ..store.migrations import initialize_database
    from ..store.repository import Repository
    from ..telemetry import normalize

    ec: dict[str, Any] = dict(experiment_context or {})
    m: dict[str, Any] = dict(metrics or {})

    if isinstance(runtime_diagnostics, RunLogParse):
        telemetry = runtime_diagnostics.telemetry
        exc_type = runtime_diagnostics.exception_type
        exc_msg = runtime_diagnostics.exception_message
        finished = runtime_diagnostics.finished
    else:
        rd: dict[str, Any] = dict(runtime_diagnostics or {})
        telemetry = normalize(rd.get("telemetry") or {})
        exc_type = rd.get("exception_type")
        exc_msg = rd.get("exception_message")
        finished = rd.get("finished", exc_type is None)

    # collect_telemetry=false -> ingest the trial without normalized telemetry (crash /
    # exception facts are kept; metric-derived classifier rules simply degrade gracefully).
    if not settings.collect_telemetry:
        telemetry = normalize({})

    status = _resolve_status(ec.get("status"), exc_type, exc_msg)

    baseline = ec.get("baseline_metric")
    post = m.get("post_change_metric")
    if post is None:
        post = m.get(settings.metric.name)
    # autoresearch writes val_bpb=0.000000 as a crash sentinel (§4) — treat as missing.
    if post == 0.0 and status in (TrialStatus.failed_oom, TrialStatus.failed_runtime):
        post = None

    if repository is None:
        initialize_database(settings)
        repository = Repository(settings)

    trial = TrialRecord(
        trial_id=new_trial_id(),
        parent_trial_id=ec.get("parent_trial_id"),
        git_commit=ec.get("git_commit") or ec.get("commit"),
        config_hash=ec.get("config_hash"),
        seed=ec.get("seed"),
        status=status,
        metric_name=settings.metric.name,
        metric_direction=settings.metric.direction,
        baseline_metric=baseline,
        post_change_metric=post,
        runtime_seconds=telemetry.runtime_seconds if telemetry.runtime_seconds is not None else m.get("runtime_seconds"),
        peak_vram_gb=telemetry.peak_vram_gb,
        throughput=telemetry.throughput if telemetry.throughput is not None else m.get("throughput"),
        exception_type=exc_type,
        exception_message=exc_msg,
        code_diff=diff,
        changed_files=list(ec.get("changed_files") or (["train.py"] if diff else [])),
        changed_components=list(ec.get("changed_components") or []),
        hyperparameters=dict(ec.get("hyperparameters") or {}),
        telemetry=telemetry.model_dump(),
    )
    repository.save_trial(trial)

    ctx = ClassificationContext(
        telemetry=telemetry,
        metric_direction=settings.metric.direction,
        baseline_metric=baseline,
        post_change_metric=post,
        exception_type=exc_type,
        exception_message=exc_msg,
        finished=bool(finished),
        baseline_hyperparameters=dict(ec.get("baseline_hyperparameters") or {}),
        changed_hyperparameters=dict(ec.get("changed_hyperparameters") or ec.get("hyperparameters") or {}),
        baseline_metric_name=ec.get("baseline_metric_name"),
        post_metric_name=ec.get("post_metric_name"),
        baseline_seed=ec.get("baseline_seed"),
        post_seed=ec.get("post_seed"),
        baseline_config_hash=ec.get("baseline_config_hash"),
        post_config_hash=ec.get("post_config_hash"),
        requires_matched_seeds=bool(ec.get("requires_matched_seeds", False)),
    )
    classification = classify(ctx, settings)
    # Persist the deterministic classifier output as its own immutable record, so its
    # category/confidence/triggered-rules survive even if the LLM later rewrites the
    # hypothesis narrative.
    repository.save_classification(trial.trial_id, classification)
    hypothesis = analyze(
        classification, ctx,
        trial_id=trial.trial_id, settings=settings, repository=repository,
        code_diff_summary=_summarize_diff(diff),
        changed_components=trial.changed_components,
    )
    # Auto-plan a controlled counterfactual for plannable categories at ingestion, so the
    # C2->C3 gate (which requires a persisted plan) has something to validate later. The
    # planner returns plans only — nothing is executed.
    if settings.counterfactual_planner_enabled:
        from ..planner import plan_counterfactual

        plan = plan_counterfactual(hypothesis, settings=settings)
        if plan is not None:
            repository.save_plan(plan)
    logger.info("recorded trial %s (status=%s, category=%s)", trial.trial_id, status.value, classification.category.value)
    return trial


# --- live-hook feeder (CLI `record`) --------------------------------------------
def _git(repo_path: str, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo_path, *args],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - environment dependent
        logger.warning("git %s failed: %s", " ".join(args), exc)
        return None


# The tunable block autoresearch's agent edits (train.py:432-451, per Phase-0 report).
_TUNABLE_NAMES = frozenset({
    "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN", "TOTAL_BATCH_SIZE", "EMBEDDING_LR",
    "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS",
    "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC", "DEPTH", "DEVICE_BATCH_SIZE",
})


def _config_hash(hyperparameters: Mapping[str, Any] | None) -> str | None:
    """Stable 16-hex identity for a tunable configuration (None when empty)."""
    if not hyperparameters:
        return None
    canonical = json.dumps(dict(hyperparameters), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _parse_tunables(source: str) -> dict[str, Any]:
    """Best-effort scrape of ``NAME = value`` tunable assignments from train.py text."""
    import ast

    found: dict[str, Any] = {}
    for line in source.splitlines():
        stripped = line.strip()
        if "=" not in stripped or stripped.startswith("#"):
            continue
        name, _, rhs = stripped.partition("=")
        name = name.strip()
        if name not in _TUNABLE_NAMES:
            continue
        rhs = rhs.split("#", 1)[0].strip()
        try:
            found[name] = ast.literal_eval(rhs)
        except (ValueError, SyntaxError):
            found[name] = rhs
    return found


# --- pre-experiment inference (CLI `brief --infer-from`) ------------------------
def load_component_map(settings: Settings) -> dict[str, str]:
    """Invert the config's ``component -> [knobs]`` mapping into ``knob -> component``."""
    mapping: dict[str, str] = {}
    for component, names in settings.section("components").items():
        if isinstance(names, (list, tuple)):
            for name in names:
                mapping[str(name)] = str(component)
    return mapping


def components_for(tunable_names: Any, settings: Settings) -> list[str]:
    """The distinct components a set of tunable names belongs to (sorted, deduped)."""
    mapping = load_component_map(settings)
    return sorted({mapping[str(name)] for name in tunable_names if str(name) in mapping})


def infer_changed_tunables(
    repo_path: str | Path = ".",
    *,
    ref: str = "HEAD",
    train_py: str = "train.py",
) -> dict[str, Any]:
    """Tunables whose value differs between the working tree and ``ref``.

    This is the pre-experiment counterpart to :func:`record_from_run`: the agent has
    edited ``train.py`` but not yet committed, so the change is the working tree against
    ``HEAD``. Returns an empty mapping when the file or the git ref is unavailable —
    the caller reports that rather than guessing what changed.
    """
    working = Path(repo_path) / train_py
    if not working.is_file():
        logger.warning("cannot infer changes: %s does not exist", working)
        return {}
    ref_source = _git(str(repo_path), "show", f"{ref}:{train_py}")
    if ref_source is None:
        logger.warning("cannot infer changes: unable to read %s:%s", ref, train_py)
        return {}

    current = _parse_tunables(working.read_text(encoding="utf-8"))
    baseline = _parse_tunables(ref_source)
    return {name: value for name, value in current.items() if baseline.get(name) != value}


def record_from_run(
    *,
    settings: Settings,
    repository: Any = None,
    commit: str,
    status: str,
    run_log_text: str | None = None,
    run_log_path: str | Path | None = None,
    repo_path: str = ".",
    branch: str | None = None,
    description: str = "",
    baseline_metric: float | None = None,
    baseline_commit: str | None = None,
    seed: int = 42,
    code_diff: str | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
) -> TrialRecord | None:
    """Feed :func:`record_rejected_trial` from an autoresearch run's live artifacts."""
    from ..telemetry import parse_run_log

    if run_log_text is None and run_log_path is not None:
        run_log_text = Path(run_log_path).read_text(encoding="utf-8")
    parsed = parse_run_log(run_log_text or "")

    if code_diff is None:
        code_diff = (
            _git(repo_path, "diff", f"{baseline_commit}..{commit}", "--", "train.py")
            if baseline_commit
            else _git(repo_path, "show", commit, "--", "train.py")
        )
    if hyperparameters is None:
        source = _git(repo_path, "show", f"{commit}:train.py")
        hyperparameters = _parse_tunables(source) if source else {}

    # A stable config identity from the tunable block, so trials sharing an identical
    # configuration are recognizably the same config even across commits/seeds.
    config_hash = _config_hash(hyperparameters)
    # Parent lineage: link to the baseline trial if it was already recorded (autoresearch
    # pins seed 42, so replication is driven by commit lineage, not seed diversity — #6).
    parent_trial_id = (
        repository.trial_id_for_commit(baseline_commit)
        if (repository is not None and baseline_commit)
        else None
    )

    experiment_context = {
        "git_commit": commit,
        "parent_trial_id": parent_trial_id,
        "config_hash": config_hash,
        "seed": seed,
        "status": status,
        "description": description,
        "baseline_metric": baseline_metric,
        "hyperparameters": dict(hyperparameters or {}),
        "changed_files": ["train.py"],
        "changed_components": ["train.py"],
    }
    metrics = {"post_change_metric": parsed.summary.get("val_bpb"), "throughput": parsed.telemetry.throughput}
    return record_rejected_trial(
        experiment_context, metrics, code_diff, parsed,
        settings=settings, repository=repository,
    )


# --- offline batch adapter (Path B; best-effort, lossy) -------------------------
def ingest_results_tsv(
    *,
    settings: Settings,
    repository: Any = None,
    tsv_path: str | Path,
    repo_path: str = ".",
    include_keep: bool = False,
) -> list[TrialRecord]:
    """Backfill from a preserved ``results.tsv`` (program.md:71). Best-effort and lossy.

    Rejected commits are reset off-branch and ``run.log`` is overwritten, so diffs are
    recovered from git history only when the commit is still reachable, and per-trial
    telemetry beyond the row's val_bpb/memory is generally gone. Rows we cannot fully
    reconstruct are still recorded with what survives — nothing is silently dropped.

    Idempotent: a commit already present in the store is skipped, so re-scanning the same
    ``results.tsv`` (or backfilling commits the live hook already captured) records nothing
    twice.
    """
    if not settings.enabled:
        logger.info("failuretrace disabled; ingest_results_tsv is a no-op")
        return []
    if repository is None:
        from ..store.migrations import initialize_database
        from ..store.repository import Repository

        initialize_database(settings)
        repository = Repository(settings)

    rows_recorded: list[TrialRecord] = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            if status == "keep" and not include_keep:
                continue
            commit = (row.get("commit") or "").strip()
            if commit and repository.count_trials_for_commit(commit) > 0:
                logger.info("skipping already-recorded commit %s (idempotent backfill)", commit)
                continue
            try:
                val_bpb = float(row.get("val_bpb") or 0.0)
            except ValueError:
                val_bpb = 0.0
            try:
                memory_gb = float(row.get("memory_gb") or 0.0)
            except ValueError:
                memory_gb = 0.0
            code_diff = _git(repo_path, "show", commit, "--", "train.py") if commit else None
            experiment_context = {
                "git_commit": commit,
                "status": status,
                "description": (row.get("description") or "").strip(),
                "changed_files": ["train.py"],
            }
            metrics = {"post_change_metric": val_bpb if val_bpb else None}
            runtime_diagnostics = {
                "telemetry": {"peak_vram_gb": memory_gb or None},
                "finished": status != "crash",
            }
            trial = record_rejected_trial(
                experiment_context, metrics, code_diff, runtime_diagnostics,
                settings=settings, repository=repository,
            )
            if trial is not None:
                rows_recorded.append(trial)
    logger.info("offline backfill from %s recorded %d trial(s)", tsv_path, len(rows_recorded))
    return rows_recorded
