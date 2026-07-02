# FAILURETRACE_SPEC.md — Phased Master Specification

Provider-free, failure-aware experiment governance layer for the autoresearch
repository. Executed by Claude Code in **seven gated phases (0–6)**. Read `CLAUDE.md`
for working rules. This document is the single source of truth; when in doubt, this
spec wins over your own preferences.

---

## 0. Project Concept

Rejected ML experiments are not useless outcomes. FailureTrace converts them into
structured, uncertainty-aware failure hypotheses, stores them as reusable negative
evidence, and uses them to reduce redundant future experimentation.

FailureTrace does NOT infer causal explanations from rejected trials. It records
observations, proposes falsifiable hypotheses, and upgrades claims only through
replication and controlled counterfactual evidence. The optional local LLM is a
structured hypothesis generator, never the causal authority. Causal support is
determined by telemetry, controlled interventions, replication, and counterfactual
validation.

The system distinguishes: observed signals, plausible failure hypotheses, alternative
explanations, missing evidence, hypothesis confidence, causal support level, and
proposed counterfactual validation experiments.

Target flow:

```
autoresearch experiment runner
        ↓
experiment telemetry collection
        ↓
deterministic failure classification
        ↓
optional local LLM hypothesis analysis
        ↓
failure record persistence
        ↓
negative evidence retrieval
        ↓
counterfactual experiment recommendation
        ↓
future experiment constraints / guidance
```

### Explicit non-goals (do not build)

Multi-agent swarms; unrestricted code-editing agents; cross-task transfer experiments;
web dashboards; custom Bayesian optimizers; a full causal inference framework;
mandatory embeddings or vector databases; a custom Optuna sampler (unless trivial,
tested, and justified — default is NO).

### Permitted stack

Python 3.11+, SQLite, Pydantic v2, local filesystem, PyYAML, Pandas, Matplotlib,
pytest, requests. Optional boundaries: Optuna (not a runtime dependency unless used by
a tested adapter/demo), Ollama-compatible local endpoint, local embeddings (future,
not MVP). Forbidden: OpenAI, Anthropic, Gemini, paid vector DBs, W&B cloud, paid APIs,
hosted inference, LangChain.

---

## 1. Execution Protocol

- Phases must be executed in order. Each phase ends with its **verification gate**:
  run the listed checks, show full output, summarize files changed, list deviations,
  commit, then STOP and await user approval.
- Use plan mode / an explicit written plan at the start of each phase before writing
  code.
- If the autoresearch repository cannot be found or its structure contradicts an
  assumption in this spec, STOP and report — do not fabricate.

### Phase map and dependency order

| Phase | Scope | Key tests |
|-------|-------|-----------|
| 0 | Repo reconnaissance + integration report (no code) | — |
| 1 | Foundation: models, enums, settings, IDs, stores, DB init, acceptance-test skeletons | T5, T16 |
| 2 | Telemetry + deterministic classifier + fixtures | T1–T4, T14, T17 |
| 3 | Hypothesis generation: fallback first, then Ollama client | T6–T9 |
| 4 | Evidence layer: retrieval, guidance, replication gate, counterfactual planner | T7–T13 |
| 5 | CLI, reporting, autoresearch integration | T15, CLI checks |
| 6 | End-to-end demo + acceptance audit + deliverables | AC1–AC14 |

---

## Phase 0 — Repository Reconnaissance (no code)

**Target repository:** `<REPO_PATH — user must fill in before starting, e.g.
~/code/autoresearch or the current working directory>`. If this placeholder has not
been replaced with a real path, ask the user for it and stop.

Inspect the actual repository and produce `docs/failuretrace_integration_report.md`
containing, with **exact file paths, function/class names, and representative line
references**:

1. Where experiments are launched.
2. Where training metrics are emitted.
3. Where validation metrics are computed.
4. Where the ratchet decides acceptance vs rejection of a change.
5. How code changes are applied; how diffs can be captured.
6. How rollback occurs.
7. The **proposed integration point** for `record_rejected_trial(...)` — the exact
   function and the minimal, flag-guarded change required. If no clean insertion point
   exists, describe the thin wrapper/adapter you will build instead. Do not fabricate
   interfaces the repo does not have.
8. **Packaging decision**: whether `failuretrace/` lives as a subpackage inside the
   repo or a sibling installable package. Default preference: a top-level
   `failuretrace/` package inside the repo with its own `pyproject.toml` extras or a
   shared one — pick what fits the repo's existing packaging and justify it.
9. What telemetry the trainer already emits vs what requires an adapter, and whether
   any core-training modification is unavoidable (it should not be, beyond optional
   telemetry emission).
10. Concurrency reality: does autoresearch ever run trials in parallel? This
    determines SQLite write assumptions (see §4.1).

**Gate 0:** report reviewed and approved by the user. No code written.

---

## Phase 1 — Foundation

### Directory structure (adapt to Phase-0 packaging decision; preserve separation)

```
failuretrace/
├── __init__.py            # public API: record_rejected_trial, initialize_database, ...
├── __main__.py
├── cli.py
├── pyproject-notes: package must be importable; add pyproject.toml if sibling package
├── config/defaults.yaml
├── core/        models.py, enums.py, settings.py, ids.py
├── store/       sqlite_store.py, json_store.py, repository.py, migrations.py
├── telemetry/   schema.py, collector.py, adapters.py        (Phase 2)
├── classifier/  rules.py, classifier.py, thresholds.py      (Phase 2)
├── analyst/     ollama_client.py, prompt.py, fallback.py, service.py  (Phase 3)
├── evidence/    retrieval.py, guidance.py, summaries.py     (Phase 4)
├── planner/     counterfactual.py, interventions.py, replication.py   (Phase 4)
├── integration/ autoresearch_adapter.py, optimizer_adapter.py         (Phase 5)
├── reporting/   summary.py, failure_map.py, plots.py        (Phase 5)
└── tests/
```

### 1.1 Settings

`core/settings.py` loads `config/defaults.yaml` into a Pydantic Settings object.
Feature flags:

```yaml
enabled: true
ollama_enabled: false
collect_telemetry: true
store_raw_json: true
counterfactual_planner_enabled: true
replication_gate_enabled: true
```

All paths are **configurable with these defaults** (never hard-coded at call sites;
the in-tree locations are development defaults only, overridable via settings and the
`FAILURETRACE_DATA_DIR` environment variable):

```yaml
paths:
  data_dir: failuretrace/data          # db + trials/ live under here
  reports_dir: failuretrace/reports
```

The Settings object must expose `settings_hash()` — a stable hash of the effective
configuration (thresholds, weights, metric direction). This hash is stored with every
classification and hypothesis record so historical results remain reproducible even
if `defaults.yaml` later changes.

Metric configuration is direction-aware and central:

```yaml
metric:
  name: val_bpb
  direction: minimize   # or maximize
```

Provide one canonical helper used by ALL comparison logic (classifier, gate,
retrieval, planner, reports):

```python
def improvement(baseline: float, post: float, direction: MetricDirection) -> float:
    """Positive = improvement, negative = regression, respecting direction."""
```

Never assume larger is better. Supported examples — minimize: val loss, val BPB,
error rate; maximize: accuracy, reward, Sharpe.

### 1.2 Enums (`core/enums.py`)

- `TrialStatus`: `completed, promoted, rejected, failed_runtime, failed_oom, invalid,
  inconclusive`.
  **State semantics (make explicit in docstring):** these are terminal, mutually
  exclusive outcomes assigned at ingestion. `promoted` and `rejected` are the
  ratchet's verdicts on a run that finished; `completed` means finished but no
  ratchet verdict is available; `failed_*` means the run did not finish; `invalid`
  means the comparison itself is unusable; `inconclusive` means finished but the
  signal does not support a verdict. A trial record never transitions status after
  persistence (immutability) — a re-run is a new trial with `parent_trial_id` set.
- `FailureCategory`: `divergence, resource_pressure, runtime_failure,
  likely_instability, likely_undertraining, possible_overfitting,
  possible_over_regularization, invalid_comparison, inconclusive, unknown`.
- `CausalSupportLevel`: `C0_observation, C1_plausible_hypothesis,
  C2_replicated_effect, C3_counterfactual_supported, C4_robust_rule`.
- `HypothesisSource`: `rule_based, local_llm, rule_based_fallback`.
- `MetricDirection`: `minimize, maximize`.

### 1.3 Core models (`core/models.py`) — Pydantic, strict

**TrialRecord** (immutable after persistence):

```
trial_id, parent_trial_id, timestamp, git_commit, config_hash, seed, status,
metric_name, metric_direction, baseline_metric, post_change_metric, metric_delta,
runtime_seconds, peak_vram_gb, throughput, exception_type, exception_message,
code_diff, changed_files, changed_components, hyperparameters, telemetry
```

`metric_delta` is stored as raw `post - baseline`; interpretation always goes through
`improvement()`. `peak_vram_gb` on the trial record is **canonical** and is copied
from telemetry at ingestion when present (telemetry's copy is the raw source).

**FailureHypothesis:**

```python
class Intervention(BaseModel):
    variable: str                     # e.g. "optimizer.lr"
    action: Literal["decrease", "increase", "set", "hold"]
    target_value: float | str | None = None
    rationale: str

class CounterfactualPlanRef(BaseModel):
    plan_id: str | None = None        # links to a persisted CounterfactualPlan
    summary: str

class FailureHypothesis(BaseModel):
    hypothesis_id: str
    trial_id: str
    source: HypothesisSource
    category: FailureCategory
    observations: list[str]
    evidence: list[str]
    hypotheses: list[str]
    alternative_explanations: list[str]
    missing_evidence: list[str]
    hypothesis_confidence: float      # bounded [0, 1]
    evidence_quality: float           # bounded [0, 1]
    suggested_intervention: Intervention
    proposed_counterfactual_trial: CounterfactualPlanRef
    should_apply_soft_penalty: bool
    should_apply_hard_constraint: bool
    causal_support_level: CausalSupportLevel
    settings_hash: str
```

Do NOT name any numeric field `causal_confidence`. Claim strength lives exclusively
in `causal_support_level`; belief strength in `hypothesis_confidence`.

**Model-level validators enforce the epistemic rules:**

- Confidence and quality bounded [0, 1].
- At creation, `causal_support_level` ∈ {C0, C1} — C2+ can only be asserted by
  promotion records (see 1.4), never by a freshly ingested single trial.
- `alternative_explanations` must be non-empty unless the category is deterministic
  (`resource_pressure` with objective evidence, `runtime_failure`, `divergence` with
  NaN/Inf detected).
- `should_apply_hard_constraint=True` is invalid unless one of: (a) deterministic AND
  repeated failure, (b) a configured objective resource limit is exceeded, or
  (c) `causal_support_level >= C2`. Validation of (a)/(c) at write time uses the
  repository (the store refuses to persist a violating record); (b) is checkable in
  the model given telemetry context passed at construction.
- A single noisy performance regression must never yield a hard constraint.
- `inconclusive` evidence may produce context/soft warnings, never hard restrictions.

**PromotionRecord** (append-only; resolves immutability vs. promotion):

```
promotion_id, hypothesis_id, from_level, to_level, timestamp,
replication_group_id | None, counterfactual_trial_id | None,
supporting_trial_ids: list[str], rationale, settings_hash
```

A hypothesis's *effective* causal support level = original level overridden by the
highest valid PromotionRecord. Hypothesis records are never mutated.

**Link records** (append-only): replication evidence, validation links, counterfactual
links — keyed by `trial_id, hypothesis_id, source_trial_id, counterfactual_trial_id,
replication_group_id` as applicable.

### 1.4 Stores

- `store/sqlite_store.py`: SQLite at `<data_dir>/failuretrace.db`. Open with WAL mode
  and a busy timeout. Document the concurrency assumption from Phase 0 (single-writer
  by default; WAL makes concurrent readers safe; if autoresearch runs parallel trials,
  serialize writes through the repository).
- `store/json_store.py`: raw immutable trial JSON under `<data_dir>/trials/` (one file
  per trial, write-once; refuse overwrite).
- `store/migrations.py`: `initialize_database()` — **idempotent**, creates a
  lightweight `schema_version` table and applies numbered schema steps. No migration
  framework.
- `store/repository.py`: the only write path. Enforces append-only semantics,
  immutability (no UPDATE/DELETE on trial/hypothesis rows; promotions and links are
  inserts), and the hard-constraint write-time checks above.

### 1.5 Acceptance-test skeletons

Create `tests/test_acceptance.py` now, with one test per acceptance criterion AC1–AC14
(§9). Tests for later phases are marked `pytest.mark.skip(reason="phase N")` — they
are un-skipped as phases land. This makes the acceptance list executable rather than
self-graded prose.

**Gate 1:** `pytest failuretrace/tests` passes (T5 store round-trip SQLite+JSON, T16
`initialize_database()` idempotent — call twice, assert identical schema and no error;
plus model-validator tests). Commit.

---

## Phase 2 — Telemetry + Deterministic Classifier

### 2.1 Telemetry (`telemetry/`)

Normalized schema (`schema.py`), all fields optional — partial metrics must be
accepted gracefully:

```
train_loss_start, train_loss_end, val_metric, train_metric,
gradient_norm_mean, gradient_norm_std, gradient_norm_max, gradient_norm_cv,
loss_spike_count, nan_detected, inf_detected,
peak_vram_gb, gpu_memory_ratio, throughput, runtime_seconds,
learning_rate_history, train_val_gap, parameter_norm_summary
```

`collector.py` builds normalized records; `adapters.py` contains adapter functions
from trainer-specific logs (write the adapter for whatever autoresearch actually
emits, per the Phase-0 report — do not couple downstream code to raw trainer logs).
GPU metrics optional; never crash when CUDA is unavailable; everything must run in
CPU-only tests.

### 2.2 Classifier (`classifier/`)

Rule-based, explainable, threshold-driven. Returns:

```python
FailureClassification(
    category=..., confidence=..., observations=[...],
    triggered_rules=[...], alternative_categories=[...],
    settings_hash=...,
)
```

Rules (each a small named function in `rules.py`; thresholds injected from Settings,
never hard-coded):

| Signal | Category |
|---|---|
| NaN or Inf detected | `divergence` |
| CUDA OOM exception, or `gpu_memory_ratio >= thresholds.gpu_memory_ratio_resource_pressure` | `resource_pressure` |
| Exception during runtime (non-OOM) | `runtime_failure` |
| `gradient_norm_cv >= thresholds.gradient_norm_cv_instability` AND metric regressed (direction-aware) | `likely_instability` |
| Train loss slope near cutoff `<= thresholds.undertraining_loss_slope` while val does not improve | `likely_undertraining` |
| `train_val_gap` increased beyond `thresholds.overfitting_train_val_gap` while val worsened | `possible_overfitting` |
| Both train and val worsen after stronger regularization (detected from hyperparameter delta) | `possible_over_regularization` |
| **Invalid comparison triggers:** baseline missing; seed mismatch where design requires matched seeds; metric_name differs between baseline and post; config_hash indicates eval-set/protocol change | `invalid_comparison` |
| Finished, no rule fires, and \|improvement\| below a configured noise floor `thresholds.inconclusive_noise_floor` | `inconclusive` |
| Nothing else applies | `unknown` |

**Confidence rubric (deterministic — no arbitrary floats):** each rule declares a
confidence tier — `deterministic → 0.95`, `strong_heuristic → 0.7`,
`weak_heuristic → 0.5`, `default → 0.3` — then the value is capped by evidence
completeness: multiply by `(available_required_fields / required_fields)` for that
rule. Tiers and the cap live in `defaults.yaml` under `confidence:`. Document the
rubric in the module docstring.

`config/defaults.yaml` (extend, don't replace, the Phase-1 keys):

```yaml
thresholds:
  gpu_memory_ratio_resource_pressure: 0.98
  gradient_norm_cv_instability: 2.0
  undertraining_loss_slope: -0.01
  overfitting_train_val_gap: 0.10
  inconclusive_noise_floor: 0.001
  replication_minimum_trials: 2
  counterfactual_minimum_support: 1
confidence:
  deterministic: 0.95
  strong_heuristic: 0.7
  weak_heuristic: 0.5
  default: 0.3
```

### 2.3 Synthetic fixtures (`tests/fixtures/`)

Programmatic fixtures (factory functions, not opaque JSON blobs) for: stable
improvement, divergence, OOM, instability, undertraining, overfitting, possible
over-regularization, inconclusive noise, invalid comparison. Reused by every later
phase.

**Gate 2:** T1 (NaN/Inf → divergence), T2 (OOM → resource_pressure), T3 (high grad CV
+ regression → likely_instability), T4 (missing telemetry never crashes; degrades to
`unknown`/`inconclusive` with reduced confidence), T14 (direction handling: identical
numbers classify oppositely under minimize vs maximize), T17 (CPU-only). Full suite
green. Commit.

---

## Phase 3 — Hypothesis Generation (fallback FIRST, then Ollama)

### 3.1 `analyst/fallback.py` (build first)

Converts a `FailureClassification` + trial context into a valid `FailureHypothesis`
with `source=rule_based` (or `rule_based_fallback` when produced because the LLM path
failed). Deterministic: observations from triggered rules, alternatives from
`alternative_categories`, missing evidence from absent telemetry fields, confidence
from the rubric, causal support C0/C1 only. This path alone must satisfy the whole
pipeline — the LLM is strictly additive.

### 3.2 `analyst/ollama_client.py`

Thin client over a local Ollama-compatible endpoint. Config:

```yaml
ollama:
  base_url: http://localhost:11434
  model: llama3.1            # user-overridable
  timeout_seconds: 30
  max_retries: 1
  format: json               # request structured JSON output
```

### 3.3 `analyst/prompt.py`

The prompt provides: normalized telemetry, code-diff summary, changed components,
hyperparameter delta, deterministic classifier output, baseline/post metrics, metric
direction, runtime diagnostics. It must explicitly instruct the model to:

- not claim causality from a single experiment;
- separate observations from hypotheses;
- state alternative explanations and missing evidence;
- propose a falsifiable counterfactual experiment;
- use only evidence in the provided record — never invent telemetry, code behavior,
  or results;
- never assign causal support above C1 for a single trial;
- never recommend a hard constraint unless the evidence explicitly shows
  deterministic repeated failure;
- return ONLY JSON matching the provided schema (embed the JSON schema exported from
  the Pydantic model).

### 3.4 `analyst/service.py`

Orchestrates: if `ollama_enabled` is false → fallback path directly. If enabled:
call Ollama; on unavailability, timeout, invalid JSON, or Pydantic validation failure
→ log structured warning, produce fallback hypothesis with
`source=rule_based_fallback`, persist it, **never crash the pipeline**. Even a valid
LLM response is re-validated by the Pydantic model (so LLM output can never smuggle
in C2+ or an unjustified hard constraint).

**Gate 3:** T6 (Ollama absent → safe fallback persisted; simulate via unroutable URL
and via a mock returning garbage JSON), T7 (single-trial hypothesis capped at C1 —
including when a mocked LLM tries to return C3), T8 (inconclusive → no hard
constraint), T9 (single OOM → no persistent hard constraint unless configured
objective limit exceeded; test both branches). No test may require a running Ollama.
Full suite green. Commit.

---

## Phase 4 — Evidence Layer

### 4.1 Retrieval (`evidence/retrieval.py`)

No vector DB, no mandatory embeddings. Deterministic, explainable structured
retrieval:

```python
def retrieve_relevant_failures(
    intervention_context: InterventionContext, top_k: int = 5,
) -> list[RetrievedFailure]:
    ...

class RetrievedFailure(BaseModel):
    hypothesis: FailureHypothesis
    relevance_score: float
    score_explanation: list[str]   # one human-readable line per scoring component
```

Weighted score over: matching failure category, matching changed component, overlap
in changed hyperparameter names, hyperparameter range proximity, effective causal
support level, recency, repeated-support count. **Weights live in `defaults.yaml`:**

```yaml
retrieval:
  weights:
    category_match: 3.0
    component_match: 2.0
    hyperparameter_overlap: 1.5
    range_proximity: 1.0
    causal_support: 2.0
    recency: 0.5
    repeated_support: 1.5
  recency_half_life_days: 30
  log_scale_parameters: [lr, learning_rate, weight_decay]  # proximity in log-space
```

Range proximity: normalized distance; use log-space for parameters listed in
`log_scale_parameters` (a 3e-4 vs 3e-3 LR gap is large; 3e-4 vs 4e-4 is small).
Every component contributing to a score appends a line to `score_explanation`.

### 4.2 Guidance (`evidence/guidance.py`, `evidence/summaries.py`)

```python
class SearchGuidance(BaseModel):
    soft_penalties: list[dict]
    hard_constraints: list[dict]
    warnings: list[str]
    relevant_failure_hypotheses: list[str]
```

Behavior: repeated instability in similar trials → warning + soft penalty; repeated
deterministic OOM → hard resource constraint; inconclusive → context only, no
constraint. **Soft penalties are the default; hard constraints only for deterministic
repeated resource failures or C2+ evidence.** Do not return raw full history —
`summaries.py` produces a compact context summary suitable for future agent prompts.

### 4.3 Counterfactual planner (`planner/counterfactual.py`, `interventions.py`)

Deterministic. Generates one controlled validation experiment per selected category:

- `likely_instability`: hold code diff + effective batch fixed; change ONE primary
  variable (reduce LR **or** increase warmup ratio).
- `likely_undertraining`: hold architecture + optimizer fixed; increase training
  budget/schedule horizon.
- `possible_overfitting`: hold architecture fixed; adjust ONE regularization variable.
- `resource_pressure`: reduce batch size or sequence length; preserve effective batch
  via gradient accumulation where possible.

Plan schema (persisted, append-only):

```
plan_id, hypothesis_id, primary_intervention_variable,
optional_coupled_stabilization_variable | None,
control_variables, treatment_variables, held_constant_variables,
expected_outcome_if_hypothesis_true, expected_outcome_if_hypothesis_false,
interaction_rationale | None, settings_hash
```

Rules: exactly one primary variable by default; a coupled intervention (two
variables) is allowed ONLY when the hypothesis explicitly concerns their interaction
(e.g., LR × warmup) and `interaction_rationale` states why both are necessary and
which interaction is tested — the model validator rejects a coupled plan without it.
Expected outcomes must be expressed direction-aware via `improvement()`. The planner
**returns plans only — it never executes experiments.**

### 4.4 Replication gate (`planner/replication.py`)

Deterministic promotion logic producing `PromotionRecord`s (never mutating
hypotheses):

- **C1 → C2**: same intervention family observed across
  `>= thresholds.replication_minimum_trials` distinct seeds or equivalent controlled
  trials (linked by `replication_group_id`).
- **C2 → C3**: a planned counterfactual trial produced the expected **directional**
  result, judged via `improvement()` under the configured metric direction, with
  `>= thresholds.counterfactual_minimum_support` supporting counterfactuals.
- **C3 → C4** (`C4_robust_rule`, rare by design): at least
  `thresholds.c4_minimum_counterfactuals: 2` independent counterfactual confirmations
  from `>= 2` distinct contexts (different `changed_components` or configs). If this
  is never reached in practice, that is correct behavior — C4 must be reachable in
  code and tests but is expected to be rare.
- The gate structurally prevents any single trial from producing C2+.
- No Bayesian causal estimation in the MVP.

**Gate 4:** T7 (gate refuses C2 from one trial), T10 (planner holds unrelated
variables constant — assert `held_constant_variables` covers everything not
intervened on), T11 (coupled plan without interaction rationale is rejected), T12
(relevant records rank above irrelevant — construct one matching and one
non-matching fixture), T13 (score explanations present and non-empty), T14 extended
(directional counterfactual result respected for both minimize and maximize). Full
suite green. Commit.

---

## Phase 5 — CLI, Reporting, Integration

### 5.1 CLI (`cli.py`, `__main__.py`)

```
python -m failuretrace init
python -m failuretrace ingest <trial.json>        # synthetic/demo ingestion
python -m failuretrace report summary
python -m failuretrace report failures
python -m failuretrace report trial <trial_id>
python -m failuretrace report map
```

### 5.2 Reporting (`reporting/`)

Matplotlib only. Artifacts under `<reports_dir>/`. Content: trial counts by status;
failure category distribution; repeated failure patterns; **effective** causal
support level distribution (post-promotion); rejection causes; soft constraint
recommendations; hard resource constraints; failure maps (grouped tables / heatmaps /
scatter — simple is fine); confidence summaries. Reports must visibly separate:
observations / plausible hypotheses / replicated effects / counterfactual-supported
effects / robust rules — and must never present C0/C1 findings as causal conclusions
(section headers should say e.g. "Plausible hypotheses (NOT causally validated)").

### 5.3 Autoresearch integration (`integration/autoresearch_adapter.py`)

Implement exactly the integration point approved in Phase 0. Public API:

```python
failuretrace.record_rejected_trial(
    experiment_context=..., metrics=..., diff=..., runtime_diagnostics=...,
)
```

— adapted to whatever shape the real repo provides (do not force this signature if
the repo's architecture differs; document the actual one). The hook in autoresearch
must be minimal and flag-guarded:

- When `enabled: false`: **zero** behavior change — no experiment, training, or
  ratchet behavior changes; no Ollama dependency; no failure record required; ideally
  the hook is a guarded no-op that never imports failuretrace internals.
- Do not modify core training behavior except, if strictly unavoidable, minimal
  telemetry emission — and justify it in writing.
- `integration/optimizer_adapter.py`: produces `SearchGuidance` for a future
  Optuna/TPE/BO/CMA-ES consumer. No Optuna runtime dependency; no custom sampler.

Update `docs/failuretrace_integration_report.md` with the exact files modified and
diff summary.

**Gate 5:** T15 — two-part, honestly scoped: (a) automated: with `enabled: false`,
the adapter is a no-op (assert no DB/JSON writes, no imports of analyst/, identical
return values through the hook); (b) manual: run autoresearch's own existing test
suite / smoke run before and after the patch with the flag off and record identical
results in the integration report. CLI commands run successfully against demo data.
Full suite green. Commit.

---

## Phase 6 — End-to-End Demo + Acceptance Audit

1. `demo/run_demo.py` (or `python -m failuretrace demo`): ingests the synthetic
   fixture set end-to-end — telemetry → classification → fallback hypothesis →
   persistence → retrieval for a new intervention context → counterfactual plan →
   replication-gate promotion on a multi-seed synthetic group → CLI summary + map.
   Ollama disabled throughout.
2. Un-skip all remaining acceptance tests; entire suite green, CPU-only, offline.
3. Produce `docs/failuretrace_deliverables.md` containing: architecture summary;
   files added/modified; exact install commands; exact init commands; exact test
   commands; the demo command; the actual ratchet integration point; known
   limitations; suggested next phase; and a short explanation of how causal support
   levels are upgraded through replication and counterfactual evidence.

**Gate 6 — Acceptance criteria (all executable as tests where marked):**

- AC1 synthetic rejected trial ingested through the public API ✅test
- AC2 normalized telemetry record produced ✅test
- AC3 deterministic classifier returns explainable category ✅test
- AC4 fallback hypothesis persisted when Ollama disabled ✅test
- AC5 trial data written to both SQLite and JSON ✅test
- AC6 relevant prior failures retrieved for a new intervention context ✅test
- AC7 retrieval includes deterministic score explanations ✅test
- AC8 counterfactual plan generated without execution ✅test
- AC9 replication gate prevents single-trial C2+ ✅test
- AC10 metric direction respected in regression/improvement logic ✅test
- AC11 CLI summary and failure-map reports run ✅test (subprocess) + artifact check
- AC12 full pytest suite passes CPU-only ✅
- AC13 FailureTrace disabled ⇒ autoresearch unchanged ✅test (a) + manual (b)
- AC14 no paid provider/cloud/hosted dependency ✅test: assert forbidden packages
  absent from the dependency set and no network calls outside localhost in tests

---

## Cross-Cutting Invariants (apply in every phase)

1. Direction-aware comparisons everywhere, via the single `improvement()` helper.
2. Append-only persistence; immutability after write; promotions/links as new records.
3. `settings_hash` stored on every classification, hypothesis, plan, and promotion.
4. Reproducibility fields on every trial: config hash, seed, git commit, metric
   direction, diff metadata.
5. Confidence values bounded [0, 1]; confidence from the rubric, never ad-hoc.
6. Single trial ⇒ C0/C1 only. Hard constraints only under the three permitted
   conditions. Inconclusive ⇒ context only.
7. Deterministic rules preferred over LLM inference whenever deterministic evidence
   exists. Never make claims beyond available evidence — in code, reports, or your
   own phase summaries.
8. CPU-only, offline test suite. Ollama, GPUs, and Optuna are always optional.
9. Structured logging; no hidden global state; no machine-specific paths.
