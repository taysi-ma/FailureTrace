# FailureTrace

[![CI](https://github.com/taysi-ma/FailureTrace/actions/workflows/ci.yml/badge.svg)](https://github.com/taysi-ma/FailureTrace/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Failure-aware experiment governance for autonomous ML research loops — provider-free,
deterministic-first, and epistemically honest.**

FailureTrace turns rejected and crashed training runs into structured, uncertainty-aware
failure hypotheses, stores them as append-only negative evidence, and uses that evidence
to guide future experiments. It is built around
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the overnight loop in
which an AI agent edits `train.py`, trains for five minutes, and keeps or discards the
change — but the ingestion API is a plain-mapping contract that any experiment runner can
call.

The founding constraint: **a single rejected trial never becomes a causal conclusion.**
Every hypothesis carries an explicit causal support level (C0–C4) that can only be raised
through replication and controlled counterfactual evidence. That rule is enforced by
Pydantic validators, the storage write path, and database triggers — not by convention.

---

- [Why](#why)
- [How it works](#how-it-works)
- [The causal support ladder](#the-causal-support-ladder)
- [Quick start](#quick-start)
- [Recording real autoresearch runs](#recording-real-autoresearch-runs)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Failure taxonomy](#failure-taxonomy)
- [Retrieval and guidance](#retrieval-and-guidance)
- [Configuration](#configuration)
- [Persistence and integrity](#persistence-and-integrity)
- [Project layout](#project-layout)
- [Development](#development)
- [Known limitations](#known-limitations)
- [Documentation](#documentation)

## Why

Autonomous experiment loops produce failures at machine speed and then throw them away.
An overnight autoresearch session runs ~100 five-minute experiments; a discarded change
leaves behind one untracked TSV row, a log that the next run overwrites, and a commit
that gets reset off the branch. Nothing stops the agent from re-running a close variant
of the same dead end tomorrow — and nothing distinguishes *"this diverged, twice, with
NaNs"* from *"this looked 0.3% worse on a single noisy run."*

FailureTrace is the memory and epistemics layer for those failures:

- **Deterministic first.** A rule-based, threshold-driven classifier explains every
  verdict (`triggered_rules`, `observations`, rubric-derived confidence). An optional
  local LLM (Ollama) may enrich the narrative, but it can never change the category,
  raise the causal level, force a constraint, or overwrite the rubric confidence — and
  any LLM failure degrades transparently to the deterministic path.
- **Honest by construction.** Claim strength lives only in `causal_support_level`
  (C0–C4); belief strength lives only in `hypothesis_confidence` [0, 1]. Fresh
  hypotheses are capped at C1. Promotions are append-only records produced by
  deterministic gates and re-verified at write time, so a causal upgrade cannot be
  forged by any caller.
- **Provider-free.** Core dependencies are `pydantic` and `pyyaml`. No OpenAI, no
  Anthropic, no cloud services, no vector DBs, no hosted inference — the test suite
  asserts the forbidden dependencies are absent and makes no non-localhost network
  calls. Everything runs CPU-only and offline.
- **Zero-touch integration.** autoresearch's `train.py` and `prepare.py` are never
  modified. With `enabled: false`, every entry point is a guarded no-op and the host
  repo behaves byte-for-byte identically.

## How it works

```
autoresearch loop (program.md)                 [or any experiment runner]
   │   run.log · results.tsv row · git diff
   ▼
record_rejected_trial(context, metrics, diff, diagnostics)
   ▼
telemetry normalization                        failuretrace/telemetry
   ▼
deterministic classifier                       failuretrace/classifier
   ▼
failure hypothesis (rule-based;                failuretrace/analyst
optional local-LLM narrative)
   ▼
append-only persistence                        failuretrace/store
(SQLite WAL + write-once JSON)
   ▼
structured retrieval + search guidance         failuretrace/evidence
   ▼
counterfactual planner + promotion gates       failuretrace/planner
   ▼
CLI · markdown reports · optional plots        failuretrace/cli, reporting
```

Every stage is deterministic and explainable; every persisted record carries a
`settings_hash` of the effective configuration so historical results stay
interpretable after thresholds change.

## The causal support ladder

| Level | Meaning | How it is reached |
|---|---|---|
| `C0_observation` | Recorded signals, nothing claimed | Ingestion |
| `C1_plausible_hypothesis` | Falsifiable hypothesis with alternatives and missing evidence | Ingestion — the **hard cap for any single trial** |
| `C2_replicated_effect` | Same intervention family failed the same way repeatedly | Gate: ≥ `replication_minimum_trials` distinct (seed, commit) units, consistent metric direction, effect above the noise floor |
| `C3_counterfactual_supported` | A planned, controlled counterfactual produced the predicted directional result | Gate: persisted `CounterfactualPlan` + ≥ `counterfactual_minimum_support` confirming trials |
| `C4_robust_rule` | Held across distinct contexts — rare by design | Gate: ≥ `c4_minimum_counterfactuals` confirmations from ≥ 2 distinct contexts |

Promotion mechanics:

- Hypotheses are immutable. A promotion is an append-only `PromotionRecord`; a
  hypothesis's *effective* level is its original level overridden by the highest valid
  promotion. `failuretrace gate` (or `advance_promotions()`) walks the whole ladder
  idempotently, writing replication/validation/counterfactual `LinkRecord`s so the full
  C1→C4 lineage is inspectable.
- The write path re-derives the evidence for every promotion: the hypothesis and all
  supporting trials must exist, `from_level` must equal the current effective level (no
  skipped rungs), and C2 requires the configured minimum of distinct supporting trials.
- Hard constraints are permitted only for (a) deterministic **and** repeated failure,
  (b) an objectively exceeded configured resource limit, or (c) effective level ≥ C2.
  Inconclusive evidence yields context and warnings — never restrictions.

## Quick start

Requires Python ≥ 3.11. No GPU, no network, no LLM needed.

```bash
git clone https://github.com/taysi-ma/FailureTrace.git
cd FailureTrace

pip install -e .                          # core: pydantic + pyyaml
pip install -e ".[analysis,ollama,dev]"   # + pandas/matplotlib (plots), requests (Ollama), pytest

failuretrace init                         # create/upgrade the database (idempotent)
failuretrace demo                         # end-to-end synthetic walkthrough (Ollama disabled)
pytest                                    # 161 tests, CPU-only, offline
```

The demo ingests eight synthetic trials, promotes a three-seed instability group to C2
through the replication gate, retrieves prior failures for a new intervention idea,
plans a counterfactual, and writes the reports:

```
FailureTrace end-to-end demo (Ollama disabled)
==============================================
trials ingested          : 8
category distribution    : {'likely_instability': 3, 'resource_pressure': 1, ...}

Replication gate (multi-seed instability group):
  supporting trials      : 3 (distinct seeds)
  effective causal level : C2_replicated_effect (single trials remain C0/C1)

Retrieval for a new instability idea: 5 relevant prior failure(s)
- [likely_instability] Optimization was unstable (high gradient-norm variability)
  and the metric regressed. (score=9.69, C2_replicated_effect)
  ...

Counterfactual plan      : plan_… (intervene on MATRIX_LR; not executed)
Search guidance          : 2 soft penalty(ies), 1 hard constraint(s)

Reports written:
  - …/reports/summary.md
  - …/reports/failure_map.md
```

## Recording real autoresearch runs

autoresearch has **no callable ratchet** — the accept/reject decision lives in
`program.md`: the agent edits `train.py`, runs it, greps `val_bpb` from `run.log`,
appends a `results.tsv` row, and keeps or `git reset`s. Rejected evidence is ephemeral
(commits reset, logs overwritten, the TSV untracked), so FailureTrace integrates as a
thin, flag-guarded adapter over the artifacts the loop actually produces
(see [docs/failuretrace_integration_report.md](docs/failuretrace_integration_report.md)
for `file:line` anchors):

- **Live hook (primary, complete capture).** `render_program_md_hook(settings)` emits an
  optional `program.md` section instructing the agent to record each rejection *before*
  the reset destroys the evidence:

  ```bash
  failuretrace record --commit <sha> --status discard --run-log run.log \
      --repo . --branch autoresearch/<tag> --description "wider MLP, higher LR"
  ```

  `record_from_run()` parses the `run.log` summary block and tracebacks, captures the
  `git diff`, scrapes the tunable block from `train.py@commit` for a `config_hash`, and
  ingests the trial plus its deterministic hypothesis.
- **Offline backfill (best-effort, lossy).** `ingest_results_tsv()` reconstructs what it
  can from a preserved `results.tsv` and the working tree. Idempotent — re-ingesting the
  same commits cannot manufacture "repeated" evidence.
- **Kill switch.** With `enabled: false` the hook renders to nothing and
  `record_rejected_trial()` returns `None` with zero writes; autoresearch remains
  byte-for-byte unchanged (verified by acceptance test AC13).

The `autoresearch/` directory in this repo is the reconnaissance clone (pinned at
`228791f`, gitignored, never modified).

## CLI reference

Installed as `failuretrace` (also runnable as `python -m failuretrace`).

| Command | Purpose |
|---|---|
| `failuretrace init` | Create/upgrade the database — idempotent, safe to re-run |
| `failuretrace demo` | End-to-end synthetic walkthrough (Ollama disabled throughout) |
| `failuretrace ingest <trial.json>` | Ingest a synthetic/demo trial JSON |
| `failuretrace record --commit … --status discard\|crash\|keep --run-log run.log [--repo … --branch … --baseline-metric … --seed …]` | Record a live run from its artifacts |
| `failuretrace gate` | Run the promotion ladder (C1→C2→C3→C4) over accumulated evidence |
| `failuretrace guidance --category <cat> --component <c> [--top-k N]` | Print search guidance for a planned intervention |
| `failuretrace report summary\|failures\|map` | Write and print a governance report |
| `failuretrace report trial <trial_id>` | Write and print a per-trial report |

Global options (before the subcommand): `--config <yaml>`, `--data-dir`,
`--reports-dir`, `--no-ollama` (force the fully deterministic/offline path).

## Python API

Everything below is re-exported from the top-level `failuretrace` package.

```python
from failuretrace import (
    InterventionContext, Repository, advance_promotions, guidance_for,
    initialize_database, load_settings, record_rejected_trial,
    retrieve_relevant_failures,
)

settings = load_settings()            # packaged defaults + env + overrides
initialize_database(settings)         # idempotent
repository = Repository(settings)     # the only write path

# 1. Record a rejected trial (returns None when `enabled: false`).
trial = record_rejected_trial(
    experiment_context={
        "git_commit": "8c1d2ef", "seed": 42, "status": "discard",
        "baseline_metric": 1.052, "changed_components": ["optimizer"],
        "hyperparameters": {"MATRIX_LR": 0.05},
    },
    metrics={"post_change_metric": 1.081, "runtime_seconds": 300.0},
    diff=None,
    runtime_diagnostics={"telemetry": {"gradient_norm_cv": 3.2}, "finished": True},
    settings=settings, repository=repository,
)

# 2. Before trying a similar idea, ask what the failure history says.
context = InterventionContext(
    changed_components=["optimizer"],
    changed_hyperparameters={"MATRIX_LR": 0.03},
    metric_direction=settings.metric.direction,
)
for failure in retrieve_relevant_failures(context, repository=repository, settings=settings):
    print(failure.relevance_score, failure.score_explanation)

guidance = guidance_for(context, settings=settings, repository=repository)
# guidance.soft_penalties / .hard_constraints / .warnings / .relevant_failure_hypotheses

# 3. Promote hypotheses whose accumulated evidence now clears a gate.
promotions = advance_promotions(repository, settings)   # {"replication": [...], ...}
```

`guidance_for()` and `soft_penalty_terms()` (in `failuretrace.integration.optimizer_adapter`)
are shaped for a future Optuna/BO consumer; FailureTrace ships no sampler and prescribes
no optimizer behavior.

## Failure taxonomy

The classifier evaluates named rules in priority order; the first hit sets the category
and the rest become `alternative_categories`. Thresholds shown are the defaults in
[defaults.yaml](failuretrace/config/defaults.yaml) — none are hard-coded.

| Category | Trigger | Tier |
|---|---|---|
| `divergence` | NaN/Inf detected | deterministic |
| `resource_pressure` | CUDA OOM exception, or `gpu_memory_ratio ≥ 0.98` | deterministic |
| `runtime_failure` | Non-OOM exception during the run | deterministic |
| `invalid_comparison` | Missing baseline; metric/seed/config mismatch between baseline and post | deterministic |
| `likely_instability` | `gradient_norm_cv ≥ 2.0` **and** the metric regressed | strong heuristic |
| `likely_undertraining` | Train loss still falling at cutoff while val did not improve | weak heuristic |
| `possible_overfitting` | `train_val_gap ≥ 0.10` while val worsened | weak heuristic |
| `possible_over_regularization` | Regularization strengthened and both train and val worsened | weak heuristic |
| `inconclusive` | Finished, no rule fired, \|improvement\| below the noise floor | — |
| `unknown` | Nothing else applies | — |

Confidence is never an ad-hoc float: `confidence = tier value × evidence completeness`,
where tiers (`deterministic 0.95`, `strong 0.7`, `weak 0.5`, `default 0.3`) come from the
config and completeness is the fraction of the rule's required telemetry fields that were
actually present. Missing telemetry degrades confidence — it never crashes the pipeline.

All comparisons are direction-aware through one canonical helper,
`improvement(baseline, post, direction)` — the default metric is autoresearch's `val_bpb`
(minimize), but `metric.direction: maximize` flips every classifier, gate, planner, and
report consistently.

## Retrieval and guidance

Retrieval is structured and deterministic — no embeddings, no vector DB. Each candidate's
score is a weighted sum of category match, changed-component overlap, hyperparameter-name
overlap, range proximity (log-space for `lr`-like parameters), effective causal support,
recency decay, and repeated-support count — and each component appends one
human-readable line:

```
5.28  likely_instability
   component overlap ['optimizer'] (jaccard=1.00): +2.00
   hyperparameter overlap ['MATRIX_LR'] (jaccard=1.00): +1.50
   hyperparameter range proximity (avg=0.78): +0.78
   causal support C1_plausible_hypothesis (rank 1/4): +0.50
   recency (age=0.0d, decay=1.00): +0.50
```

Guidance built on top of retrieval is **soft by default**: repeated instability near your
planned change yields warnings and soft penalties; only repeated deterministic failures
(distinct commits, not re-ingested duplicates) or C2+ evidence produce hard constraints;
inconclusive history is surfaced as context only.

## Configuration

All flags, thresholds, weights, and paths live in
[failuretrace/config/defaults.yaml](failuretrace/config/defaults.yaml) and load through a
single `Settings` object. Precedence: packaged defaults → `FAILURETRACE_DATA_DIR` env var
→ explicit overrides (`--config`, `--data-dir`, `--reports-dir`, `--no-ollama`, or the
`overrides=` argument to `load_settings()`).

| Group | Keys (defaults) | Purpose |
|---|---|---|
| Flags | `enabled: true`, `ollama_enabled: true`, `collect_telemetry`, `store_raw_json`, `counterfactual_planner_enabled`, `replication_gate_enabled` | Master kill switch and per-stage toggles |
| `metric` | `name: val_bpb`, `direction: minimize` | Direction-aware comparison everywhere |
| `paths` | `data_dir: failuretrace/data`, `reports_dir: failuretrace/reports` | DB, raw trial JSON, and report locations |
| `thresholds` | classifier triggers, `inconclusive_noise_floor`, `replication_minimum_trials: 2`, `counterfactual_minimum_support: 1`, `c4_minimum_counterfactuals: 2`, `resource_vram_limit_gb` | Every classifier and gate limit |
| `confidence` | `deterministic: 0.95` … `default: 0.3` | The confidence rubric tiers |
| `retrieval` | component weights, `recency_half_life_days: 30`, `log_scale_parameters`, `min_relevance_score` | Scoring weights and cutoffs |
| `ollama` | `base_url: http://localhost:11434`, `model: llama3.1`, timeout/retries | Optional local LLM endpoint |

`ollama_enabled` ships **on** because the LLM is strictly additive and fails closed onto
the deterministic path — but with no endpoint running you pay a timeout before fallback,
so use `--no-ollama` (or set it `false`) for fully offline runs. A `settings_hash` of the
semantic configuration is stamped on every classification, hypothesis, plan, and
promotion.

## Persistence and integrity

- **SQLite** at `<data_dir>/failuretrace.db` — WAL mode, 5 s busy timeout, foreign keys
  enforced. Single-writer assumption (matching autoresearch's serial loop); WAL keeps
  concurrent readers safe and writes serialize through the repository.
- **Write-once JSON sidecars** under `<data_dir>/trials/` — one immutable file per
  trial, atomic write, overwrite refused. SQLite is written first (authoritative);
  `reconcile_json()` restores missing sidecars after a crash between the two writes.
- **Idempotent migrations** (`initialize_database()`), schema v1–v5: base tables →
  plans → foreign keys → **immutability triggers** (the database itself rejects
  UPDATE/DELETE on trials, hypotheses, promotions, plans, and links) → persisted
  classifier provenance.
- **`Repository` is the only write path.** It enforces append-only semantics, the
  hard-constraint conditions, and the promotion evidence gate described above.

## Project layout

```
failuretrace/
├── core/          # Pydantic models + epistemic validators, enums, Settings, improvement(), IDs
├── config/        # defaults.yaml — every flag, threshold, and weight
├── telemetry/     # normalized TelemetryRecord, collector, run.log parser
├── classifier/    # deterministic rules, confidence rubric, thresholds
├── analyst/       # rule-based hypothesis builder + optional Ollama client
├── store/         # SQLite (WAL) store, write-once JSON store, migrations, Repository
├── evidence/      # structured retrieval, search guidance, compact summaries
├── planner/       # counterfactual plans, replication/counterfactual/C4 gates
├── integration/   # autoresearch adapter (live hook + offline backfill), optimizer adapter
├── reporting/     # summary / failure-map / per-trial reports, optional matplotlib plots
├── tests/         # 161 tests incl. the executable acceptance suite AC1–AC14
├── cli.py         # `failuretrace` / `python -m failuretrace`
└── demo.py        # end-to-end synthetic walkthrough
demo/run_demo.py   # demo entry point without installing the package
docs/              # Phase-0 integration reconnaissance + deliverables
autoresearch/      # reconnaissance clone (gitignored, pinned, never modified)
```

Reports are markdown-first (`summary.md`, `failure_map.md`, `trial_<id>.md`) and visibly
separate epistemic strata — "plausible hypotheses (NOT causally validated)" versus
replicated and counterfactual-supported effects. PNG charts are added only when
matplotlib is installed.

## Development

```bash
pip install -e ".[analysis,ollama,dev]"
pytest            # 161 passed — CPU-only, offline, no Ollama required
```

CI runs the suite on Python 3.11–3.13 plus a packaging smoke job that installs the wheel
and drives the CLI from a foreign working directory. The acceptance criteria AC1–AC14
from the [specification](FAILURETRACE_SPEC.md) are executable tests in
[test_acceptance.py](failuretrace/tests/test_acceptance.py), covering ingestion,
classification, dual-store persistence, retrieval explanations, planner non-execution,
the single-trial C2 ban, direction handling, CLI artifacts, the disabled no-op, and the
provider-free guarantee.

## Known limitations

- Rejected trials are ephemeral in autoresearch, so the offline TSV backfill is
  best-effort; the live `program.md` hook is the complete capture path.
- autoresearch emits only a summary block and tracebacks — gradient/loss-spike/LR-history
  rules degrade gracefully rather than firing.
- `train.py` pins seed 42; replication therefore counts distinct (seed, commit) units,
  so independent runs on different commits replicate while deterministic re-runs of one
  commit correctly do not.
- The adapters are validated against realistic CPU stand-in artifacts (real OOM
  tracebacks, `FAIL` markers, a `results.tsv` corpus); a live GPU run against the real
  loop is the one unshipped validation.
- The undertraining/overfitting heuristics use whole-run proxies pending richer
  telemetry; tuning is deferred to real data rather than guessed against fixtures.

## Documentation

| Document | Contents |
|---|---|
| [FAILURETRACE_SPEC.md](FAILURETRACE_SPEC.md) | The phased master specification (phases 0–6, tests T1–T17, acceptance criteria) |
| [docs/failuretrace_integration_report.md](docs/failuretrace_integration_report.md) | Phase-0 reconnaissance of autoresearch with `file:line` anchors; integration decisions |
| [docs/failuretrace_deliverables.md](docs/failuretrace_deliverables.md) | Deliverables summary: architecture, commands, promotion semantics, limitations |

## License

MIT
