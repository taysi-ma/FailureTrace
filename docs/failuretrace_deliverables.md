# FailureTrace — Deliverables (Phase 6)

Provider-free, failure-aware experiment governance for the **autoresearch** repository.
Turns rejected/crashed ML experiments into structured, uncertainty-aware failure
hypotheses, stores them as reusable negative evidence, and uses them to guide future
experimentation — **without ever claiming causality from a single trial**.

Status: Phases 0–6 complete, plus post-audit hardening — **P0** (evidence-checked
promotion gate, idempotent ingestion, database foreign keys) and **P1** (wired governance
loop with `gate`/`guidance` CLI + auto-planning, commit-based replication key, explicit
replication links, physical immutability triggers, LLM-confidence provenance). Test suite:
**151 passed, 0 skipped**, CPU-only, offline. All acceptance criteria **AC1–AC14** pass.

---

## 1. Architecture summary

Deterministic-first pipeline; the optional local LLM is strictly additive.

```
autoresearch run (program.md loop)
   │  results.tsv row + run.log + git diff        ← thin adapter, zero-touch
   ▼
record_rejected_trial(experiment_context, metrics, diff, runtime_diagnostics)
   ▼
telemetry (normalize / parse_run_log)            failuretrace/telemetry/
   ▼
deterministic classifier  → FailureClassification failuretrace/classifier/
   ▼
hypothesis (rule-based fallback; optional Ollama) failuretrace/analyst/
   ▼
persistence: SQLite (WAL) + append-only JSON      failuretrace/store/
   ▼
retrieval + guidance (deterministic, explainable) failuretrace/evidence/
   ▼
counterfactual planner + replication gate         failuretrace/planner/
   ▼
CLI + matplotlib-optional reports                 failuretrace/cli.py, reporting/
```

Design invariants (enforced in code + tests): direction-aware comparisons via one
`improvement()` helper; append-only, immutable-after-write persistence (promotions/links
are new records); `settings_hash` on every classification/hypothesis/plan/promotion; a
single trial can only ever be **C0/C1**; hard constraints only under three objective
conditions; confidence from a fixed rubric; everything runs CPU-only and offline.

## 2. Files

### Package (`failuretrace/`)
- `core/` — `enums.py`, `models.py` (Pydantic, epistemic validators), `settings.py`
  (`improvement()`, `settings_hash()`), `ids.py`
- `config/defaults.yaml` — every threshold / weight / flag (no magic numbers in code)
- `store/` — `sqlite_store.py` (WAL + busy_timeout; foreign keys enforced),
  `json_store.py` (write-once), `migrations.py` (idempotent, schema v1–v3; v3 adds
  referential-integrity foreign keys), `repository.py` (**the only write path**;
  hard-constraint gate; **promotion evidence gate**; effective-level computation)
- `telemetry/` — `schema.py`, `collector.py`, `adapters.py` (`parse_run_log`)
- `classifier/` — `rules.py`, `classifier.py`, `thresholds.py`, `context.py`
- `analyst/` — `fallback.py`, `ollama_client.py`, `prompt.py`, `service.py`
- `evidence/` — `retrieval.py`, `guidance.py`, `summaries.py`
- `planner/` — `interventions.py`, `counterfactual.py`, `replication.py`
- `integration/` — `autoresearch_adapter.py` (public API + hook + adapters),
  `optimizer_adapter.py` (`SearchGuidance` producer; no Optuna)
- `reporting/` — `summary.py`, `failure_map.py`, `trial.py`, `plots.py` (matplotlib-optional)
- `cli.py`, `__main__.py`, `demo.py`
- `tests/` — models, stores, telemetry, classifier, analyst, evidence, planner,
  integration, reporting, cli, demo, and `test_acceptance.py` (AC1–AC14)

### Repo root
- `pyproject.toml` (py≥3.11; `[project.scripts] failuretrace`), `demo/run_demo.py`,
  `docs/failuretrace_integration_report.md` (Phase 0 + §8 Phase 5), this file.
- `autoresearch/` — the reconnaissance target, **cloned & gitignored, never modified**.

## 3. Commands

Install (core is minimal + provider-free; extras are optional):
```
pip install -e .                       # core: pydantic, pyyaml
pip install -e ".[analysis,ollama,dev]"  # + matplotlib/pandas (reports), requests (Ollama), pytest
```

Initialize the database (idempotent):
```
python -m failuretrace init
# custom location:
python -m failuretrace --data-dir /path/to/data --reports-dir /path/to/reports init
```

Run the test suite (CPU-only, offline; no Ollama required):
```
python -m pytest failuretrace/tests        # or simply: pytest
```

Run the end-to-end demo (Ollama disabled throughout):
```
python -m failuretrace demo                 # or: python demo/run_demo.py
```

Ingest / gate / guidance / report / record:
```
python -m failuretrace ingest trial.json
python -m failuretrace gate                  # promote replicated C1 hypotheses to C2 (writes links)
python -m failuretrace guidance --category likely_instability --component optimizer
python -m failuretrace report summary       # also: failures | map | trial <trial_id>
python -m failuretrace record --commit <hash> --status <discard|crash> \
    --run-log run.log --repo . --branch autoresearch/<tag> --description "<desc>"
```

## 4. The actual autoresearch ratchet integration point

autoresearch has **no callable ratchet** — the accept/reject decision lives in
`program.md` (the agent edits `train.py`, runs `uv run train.py > run.log`, greps
`val_bpb`, appends a `results.tsv` row, and keeps or `git reset`s). See
`docs/failuretrace_integration_report.md` for `file:line` anchors. Integration is
therefore a **thin, flag-guarded, zero-touch adapter**:

- **Live hook (primary, complete capture)** — `render_program_md_hook(settings)` emits an
  optional `program.md` section (absent when `enabled: false`) instructing the agent to run
  `python -m failuretrace record …` at the moment of rejection, *before* `git reset`/log
  overwrite. `record_from_run()` parses `run.log`, captures the `git diff`, and scrapes the
  tunable block from `train.py@commit`.
- **Offline batch (best-effort, lossy)** — `ingest_results_tsv()` backfills from a preserved
  `results.tsv` + working tree (documented as lossy: reset commits / overwritten logs).
- **Disabled ⇒ no-op** — with `enabled: false` the hook is absent and
  `record_rejected_trial()` returns `None` with zero writes ⇒ autoresearch is byte-for-byte
  identical (AC13, verified: the clone stays pinned at `228791f`, clean tree).

`train.py`/`prepare.py` are **never** modified; no core training behavior changes.

## 5. How causal support levels are upgraded

Claim strength lives **only** in `causal_support_level`; belief strength in
`hypothesis_confidence`. A fresh hypothesis is capped at **C0/C1**. Upgrades are produced
**only** by the deterministic replication gate as append-only `PromotionRecord`s (the
hypothesis is never mutated); a hypothesis's *effective* level = its original level
overridden by the highest valid promotion.

- **C1 → C2** (`evaluate_replication`): the same intervention family — matched by the
  source trial's fingerprint (changed components + hyperparameter names) — reproduced
  across `≥ replication_minimum_trials` distinct seeds / equivalent controlled trials
  (linked by a `replication_group_id`), all pointing the **same** metric direction and
  clearing the noise floor. Every supporting trial must exist in the store; fabricated or
  wrong-family evidence is ignored. This structurally prevents any single trial from C2+.
- **C2 → C3** (`evaluate_counterfactual`): the hypothesis is *currently* effective-C2, a
  **persisted** `CounterfactualPlan` exists for it, and `≥ counterfactual_minimum_support`
  counterfactual trials produced the expected **directional** result above the noise floor
  (judged via `improvement()` under the configured metric direction).
- **C3 → C4** (`evaluate_c4`, rare by design): `≥ c4_minimum_counterfactuals` independent
  confirmations from `≥ 2` distinct contexts (different changed components / configs).

Every promotion is re-checked at the write path (`Repository.save_promotion`): the
hypothesis and all supporting trials must exist, `from_level` must equal the hypothesis's
*current* effective level (the ladder cannot be skipped), and a C2 promotion must carry at
least the configured minimum of distinct supporting trials — so a causal upgrade is
non-forgeable regardless of the caller. Database foreign keys (schema v3) reject dangling
references, and duplicate ingestion of one physical failure cannot manufacture "repeated"
evidence (guidance counts distinct source commits; offline backfill is idempotent).

Hard constraints are permitted only for (a) deterministic **and** repeated failure, (b) an
objectively-exceeded configured resource limit, or (c) effective level ≥ C2 — enforced at
write time by the repository. Inconclusive evidence yields context/soft warnings only.

## 6. Known limitations

- **Rejected trials are ephemeral in autoresearch** (commits reset off-branch, `run.log`
  overwritten, `results.tsv` untracked) ⇒ the offline adapter is best-effort; the live hook
  is the complete path.
- **Sparse telemetry**: autoresearch emits only the summary block, a `FAIL` marker, and
  tracebacks. Gradient/loss-spike/LR-history metrics are absent, so those classifier rules
  degrade gracefully (never crash) rather than firing.
- **Single-seed by default** (`train.py` pins seed 42): C1→C2 replication is driven by
  distinct **(seed, commit)** units, so independent runs on different commits replicate even
  at a fixed seed; deterministic re-runs of one commit correctly do not. The live adapter
  records `config_hash` + `parent_trial_id` so a group's lineage is reconstructable.
- **LLM is optional and additive**: with Ollama absent/failing, the deterministic fallback
  fully drives the pipeline; the LLM can never raise causal level, force a hard constraint,
  or overwrite the rubric confidence — its stated belief is recorded only in `llm_confidence`.
- **Reports**: matplotlib is optional — PNGs are produced only when it is installed; the
  markdown artifacts are always produced.
- `prepare.py`/`train.py` require an NVIDIA GPU + dataset to execute and were never run
  (out of the CPU-only, provider-free remit).

## 7. Suggested next phase (beyond the current spec)

- A real live-run integration test against a GPU host (record from an actual autoresearch
  loop), and a small results.tsv corpus for offline-backfill validation.
- Optional local embeddings for retrieval (kept out of the MVP; current retrieval is
  deterministic and explainable and needs no vector DB).
- A thin, tested Optuna/BO consumer of `SearchGuidance` (adapter exists; no sampler shipped).
- Multi-context C3/C4 accumulation from longer running histories to exercise the upper gate.
