# FailureTrace ↔ autoresearch — Integration Reconnaissance Report (Phase 0)

> Deliverable for **Phase 0** of `FAILURETRACE_SPEC.md`. No package code is written in
> this phase. Every file/function/line reference below resolves against the local clone at
> the pinned commit. Where the spec assumed an interface that does not exist, this report
> says so plainly rather than inventing one (per `CLAUDE.md`: no invented interfaces).

## 0. Reproducibility header

| Field | Value |
|---|---|
| Target repo | `https://github.com/karpathy/autoresearch.git` (resolves the spec's `<REPO_PATH>` placeholder) |
| Clone location | `./autoresearch` (sibling inside this `failureTrace` repo; gitignored) |
| Pinned commit | `228791fb499afffb54b46200aca536f79142f117` (`228791f`) |
| Branch | `master` |
| Remote branches observed | `origin/master`, `origin/agenthub`, `origin/exp/H100/mar8` |
| Description | "AI agents running research on single-GPU nanochat training automatically" |
| License | MIT |
| Recon date | 2026-07-02 |
| Method | `git clone` + read-only inspection of source. `prepare.py`/`train.py` were **not executed** (require an NVIDIA GPU + a HuggingFace dataset download — out of Phase-0 scope). |

## 1. Executive summary — the finding that reshapes integration

**autoresearch has no callable "ratchet," no experiment-runner API, and no code-level
accept/reject function.** The whole optimize loop is an **AI coding agent following
`program.md`** (a human-authored Markdown "skill"). The agent edits one file (`train.py`),
commits, runs it as a subprocess, greps a metric out of stdout, and **keeps or `git reset`s**
based on whether `val_bpb` dropped. The only machine-readable record of an experiment is a
row appended to `results.tsv` — which is **gitignored and explicitly kept untracked**.

Consequences for FailureTrace:

- There is **no function to wrap** with `record_rejected_trial(...)`. Integration must be a
  **thin adapter** over the artifacts the loop actually produces (`results.tsv`, `run.log`,
  `git diff`) and/or an **additive, flag-guarded instruction in `program.md`**. This is
  exactly the "write a thin adapter — never pretend an interface exists" path CLAUDE.md
  mandates.
- **Rejected trials are ephemeral.** On rejection the agent `git reset`s the commit off the
  branch (`program.md:104`); `run.log` is overwritten on the next run (`> run.log`,
  `program.md:99`); `results.tsv` is untracked/gitignored. So a purely-offline "read it
  later" adapter is **lossy**. The complete capture path is **live** (record at the moment
  of rejection, before reset/overwrite). This refines the plan's ordering — see §5.
- The metric is `val_bpb`, **lower is better ⇒ `direction: minimize`**, exactly matching the
  spec default. No contradiction there.
- **No core modification is required or warranted.** train.py already prints all available
  telemetry to stdout; we will not touch the agent-edited file. With `enabled: false`,
  FailureTrace is simply never invoked ⇒ autoresearch is byte-for-byte identical (AC13).

## 2. Repository facts

Flat repo, 3 files that matter (`README.md` "How it works"):

| File | Role | FailureTrace relevance |
|---|---|---|
| `prepare.py` (389 ln) | Fixed constants, data prep, tokenizer, dataloader, **ground-truth eval**. Read-only by contract. | Source of the metric definition + fixed budget. Never modified. |
| `train.py` (630 ln) | Full GPT model, Muon+AdamW optimizer, training loop. **The only file the agent edits.** Flat script — no `main()`, no `argparse`, no `try/except`; config = module-level constants. | Source of the diff, hyperparameters, and stdout telemetry. Never modified by us. |
| `program.md` (113 ln) | Human-authored agent instructions = the "ratchet" + logging protocol. | The real accept/reject logic and the (optional) live-hook insertion point. |
| `pyproject.toml` | `uv`-managed, py≥3.10, deps: torch 2.9.1 (cu128), pandas, numpy, matplotlib, pyarrow, requests, tiktoken, rustbpe, kernels. Agent is **forbidden to add deps** (`program.md`). | Drives the packaging decision (§3.8). |

## 3. Phase-0 required findings

### 3.1 Where experiments are launched
`uv run train.py` as a subprocess, driven by the agent per `program.md:99`
(`uv run train.py > run.log 2>&1`). There is **no Python launcher/entry point** — `train.py`
executes top-to-bottom (no `if __name__ == "__main__"`, no `main()`). Per-run seed is
hard-coded: `torch.manual_seed(42)` / `torch.cuda.manual_seed(42)` (`train.py:458–459`), so
trials are single-seed (42) unless the agent edits it — relevant to replication (§5).

### 3.2 Where training metrics are emitted
Two stdout surfaces in `train.py`:
- **Live per-step log** (`train.py:590`): `step … | loss: … | lrm: … | dt: … | tok/sec: … | mfu: … | epoch: … | remaining: …` (carriage-return progress line).
- **Final summary block** (`train.py:621–630`), the canonical machine-readable telemetry:
  ```
  ---
  val_bpb:          <float>
  training_seconds: <float>
  total_seconds:    <float>
  peak_vram_mb:     <float>
  mfu_percent:      <float>
  total_tokens_M:   <float>
  num_steps:        <int>
  num_params_M:     <float>
  depth:            <int>
  ```
- **Divergence marker** (`train.py:570–571`): `if math.isnan(train_loss_f) or train_loss_f > 100: print("FAIL")` — a deterministic NaN/blow-up signal in the log.

All telemetry is **stdout only** (captured to `run.log`); there is no JSON/CSV sidecar and no metrics callback.

### 3.3 Where validation metrics are computed
`evaluate_bpb(model, tokenizer, batch_size)` at `prepare.py:344` — the read-only ground-truth
metric, called from `train.py:613`. Uses fixed `MAX_SEQ_LEN=2048` (`prepare.py:30`) and
`EVAL_TOKENS = 40*524288` (`prepare.py:32`) "so results are comparable across configs"
(`prepare.py:350`). Budget is fixed: `TIME_BUDGET=300`s (`prepare.py:31`), enforced at
`train.py:603`.

### 3.4 Where the ratchet decides acceptance vs rejection
**Not in code — in `program.md`.** The experiment loop (`program.md:94–104`):
```
1 look at git state   2 hack train.py   3 git commit   4 uv run train.py > run.log
5 grep "^val_bpb:\|^peak_vram_mb:" run.log   6 empty grep ⇒ crash (tail run.log)
7 append row to results.tsv (do NOT commit it)
8 val_bpb improved (lower) ⇒ advance the branch, keep the commit
9 val_bpb equal/worse    ⇒ git reset back to where you started
```
The durable verdict is the `results.tsv` `status` column. Contract (`program.md:66–87`),
**tab-separated**, 5 columns:

| col | field | notes |
|---|---|---|
| 1 | `commit` | short hash, 7 chars |
| 2 | `val_bpb` | `0.000000` for crashes |
| 3 | `memory_gb` | `peak_vram_mb/1024`, `.1f`; `0.0` for crashes |
| 4 | `status` | `keep` \| `discard` \| `crash` |
| 5 | `description` | free text (no tabs) |

`keep` = ratchet accepted (val_bpb improved); `discard` = ratchet rejected (equal/worse);
`crash` = did not finish (OOM/bug/>10min timeout, `program.md:108–110`).

### 3.5 How code changes are applied; how diffs are captured
The agent edits the module-level constants / code in `train.py` directly (the tunable block
is `train.py:432–451`: `ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN, TOTAL_BATCH_SIZE,
EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR, WEIGHT_DECAY, ADAM_BETAS, WARMUP_RATIO,
WARMDOWN_RATIO, FINAL_LR_FRAC, DEPTH, DEVICE_BATCH_SIZE`), then `git commit` (`program.md:98`)
on branch `autoresearch/<tag>` (`program.md:10`, `:92`). **Diff capture** = standard git:
`git diff <parent> <commit> -- train.py` (or `git show <commit>`), while the commit still
exists on the branch or in reflog.

### 3.6 How rollback occurs
`git reset` back to the pre-experiment commit on rejection (`program.md:104`), and
"discard and revert" on timeout/broken idea (`program.md:108,110`). **This removes the
rejected commit from the branch tip** (recoverable only via `git reflog` within gc window).

### 3.7 Proposed `record_rejected_trial(...)` integration point  ← the crux

No clean code call-site exists, so FailureTrace uses a **thin adapter** with two paths. The
public API keeps the spec's shape and is fed by the adapter with real data:
```python
failuretrace.record_rejected_trial(experiment_context, metrics, diff, runtime_diagnostics)
```

**Path A — live hook (recommended; complete capture). Built in Phase 5.**
Append an *optional*, clearly-fenced section to `program.md` instructing the agent, right
after step 7 (log the row) and **before** the reset/next run, to call:
```
python -m failuretrace record --commit <hash> --status <discard|crash> \
    --run-log run.log --repo . --branch autoresearch/<tag> --description "<desc>"
```
The CLI then: parses `run.log` (summary block §3.2 + `FAIL` marker + traceback), captures
`git diff` for `<hash>`, reads the `results.tsv` row, and persists a `TrialRecord` +
fallback `FailureHypothesis`. Rationale it is *recommended over* offline: it captures
`run.log` and the diff **before** they are overwritten / reset away (§1, §3.6).
- **Flag-guarded**: the section is present only when `failuretrace.enabled: true`. Omit it ⇒
  the agent never calls FailureTrace ⇒ identical autoresearch behavior. `train.py` untouched.

**Path B — offline batch adapter (best-effort; zero-touch). `integration/autoresearch_adapter.py`, Phase 5.**
Point it at a run branch's working tree; it ingests durable artifacts that survive: a
preserved `results.tsv` (rows ⇒ commit/val_bpb/memory/status/description), diffs recoverable
from branch history or `git reflog`, and the *current* `run.log`. Useful for backfill, but
**lossy** for already-reset rejects and overwritten logs — documented as such, no silent
truncation.

Both paths require **zero** modification to `train.py`/`prepare.py` and no change to the
accept/reject logic, so AC13 (`enabled:false` ⇒ no-op) holds for free.

### 3.8 Packaging decision
**FailureTrace is a sibling package inside this `failureTrace` repo**, beside the gitignored
clone:
```
failureTrace/
├── FAILURETRACE_SPEC.md  CLAUDE.md  README.md
├── pyproject.toml                 (NEW, Phase 1 — failuretrace, py≥3.11)
├── failuretrace/                  (NEW, Phase 1+ — package)
├── autoresearch/                  (CLONED @228791f, gitignored)
└── docs/failuretrace_integration_report.md
```
Rationale: autoresearch forbids adding dependencies to its own `pyproject.toml`
(`program.md`), so FailureTrace ships its own deps (pydantic v2, pyyaml, pandas, matplotlib,
pytest, requests) in a separate distribution, keeping autoresearch pristine. Nesting
`failuretrace/` *inside* the gitignored clone (the spec's nominal default) would leave the
package untracked by the repo the user actually works in — so the sibling layout is the
correct fit here. FailureTrace targets **py≥3.11** (spec) vs autoresearch's 3.10 pin;
harmless as separate venvs (the offline adapter only needs stdlib + pandas, not torch).

### 3.9 What telemetry the trainer already emits vs what needs an adapter

| FailureTrace telemetry field (spec §2.1) | Source in autoresearch | Adapter work |
|---|---|---|
| `val_metric` (val_bpb) | `train.py:622` summary | parse stdout |
| `runtime_seconds` | `training_seconds` `train.py:623` (≈300, fixed budget) | parse |
| `peak_vram_gb` | `peak_vram_mb` `train.py:625` ÷1024 | parse ÷1024 |
| `throughput` | derive from `total_tokens_M` `:627` / `training_seconds`, or `mfu_percent` `:626` | derive |
| `nan_detected` / `inf_detected` | `FAIL` marker `train.py:571` (NaN or loss>100) | grep marker ⇒ divergence |
| `exception_type` / `exception_message` | Python traceback in `run.log` on crash (`program.md:101`) | parse traceback |
| `train_loss_start/end` | live log `train.py:590` (per-step, not in summary) | optional log-scrape |
| `gradient_norm_*`, `loss_spike_count`, `gpu_memory_ratio`, `learning_rate_history`, `train_val_gap`, `parameter_norm_summary` | **NOT emitted** | absent ⇒ classifier degrades gracefully (Phase 2 **T4**); **no core change** to add them |

`hyperparameters` come from parsing the tunable block (`train.py:432–451`) at the trial's
commit; `changed_files` = `["train.py"]` by construction; `code_diff`/`changed_components`
from the git diff. `gpu_memory_ratio` is not directly computable (no total-VRAM emitted) —
adapter may leave it unset or derive from a configured device capacity in Settings.

### 3.10 Concurrency reality
**Parallel trials are real.** `program.md:92` names per-GPU branches
(`autoresearch/mar5-gpu0`); autoresearch's `.gitignore` lists `worktrees/`, `results/`,
`queue/` (launcher-managed, one worktree per concurrent agent), and `origin/agenthub` is a
launcher branch. ⇒ FailureTrace must assume **concurrent writers**: SQLite opened in **WAL**
mode with a **busy_timeout**, and **all writes serialized through the repository**
(spec §4.1). Readers stay safe under WAL.

## 4. Data mapping: autoresearch → FailureTrace models

| autoresearch artifact | FailureTrace field / enum |
|---|---|
| `results.tsv` status `keep` | `TrialStatus.promoted` (ratchet accepted) |
| status `discard` | `TrialStatus.rejected` (finished, worse/equal) |
| status `crash` + CUDA-OOM traceback | `TrialStatus.failed_oom` ⇒ `FailureCategory.resource_pressure` |
| status `crash` + other traceback | `TrialStatus.failed_runtime` ⇒ `FailureCategory.runtime_failure` |
| `FAIL` marker (NaN/loss>100) | `FailureCategory.divergence` |
| `val_bpb=0.000000` sentinel on crash | treat as *missing metric*, **not** a real 0 bpb |
| commit hash | `TrialRecord.git_commit`; short hash links the row |
| seed 42 (`train.py:458`) | `TrialRecord.seed` (constant unless edited) |
| `val_bpb`, direction=minimize | `metric_name`/`metric_direction`; deltas via `improvement()` |

## 5. Deviations & refinements from spec / approved plan

1. **Ratchet is not code.** Accept/reject lives in `program.md` + `results.tsv` + git, not a
   function ⇒ integration is a thin adapter (§3.7), not a code-site hook. (Spec anticipated
   this fallback; recorded here as fact, not fabrication.)
2. **Rich telemetry mostly absent** (§3.9). Only the summary block, `FAIL` marker, and
   tracebacks exist. Classifier must degrade gracefully; **no** core modification to add
   gradient/loss-spike metrics. Matches Phase-2 T4 intent.
3. **Recommendation order flipped vs the approved plan.** The plan named the *offline*
   adapter "primary/default" and the live hook "secondary." Recon shows offline is **lossy**
   (rejected commits reset off-branch `program.md:104`; `run.log` overwritten `:99`;
   `results.tsv` gitignored/untracked `:102`). Honest conclusion: the **live hook is the
   complete/primary path**, offline is best-effort backfill. Both remain flag-guarded no-ops
   when disabled. Flagged for your call at Gate 0.
4. **Replication caveat.** Spec §4.4 C1→C2 wants "distinct seeds." autoresearch pins seed 42
   (`train.py:458`); replication here will more naturally come from repeated/independent runs
   on separate branches than from seed sweeps. To be handled by the replication gate design
   in Phase 4 (linking via `replication_group_id`); noted now so it isn't a surprise.
5. **Python 3.11 (FailureTrace) vs 3.10 (autoresearch)** — separate venvs; non-issue.

## 6. Out of scope / not done in Phase 0
- Did **not** run `prepare.py` or `train.py` (need an NVIDIA GPU + HF `climbmix` dataset
  download; also outside FailureTrace's provider-free, CPU-only test remit).
- Wrote **no** package/module code and modified **no** autoresearch file. Only this report
  and a `.gitignore` update land in this phase.

## 7. Gate 0 checklist
- [x] Target repo resolved, cloned, and pinned (`228791f`).
- [x] All 10 Phase-0 items answered with real `file:line` anchors.
- [x] Integration point proposed as a thin adapter — no invented interface.
- [x] Packaging decision made and justified.
- [x] Concurrency reality established (WAL + serialized writes).
- [x] Deviations/refinements recorded (§5), incl. the offline-vs-live refinement for review.
- [x] No package code written; no autoresearch file modified.

**Awaiting user approval to proceed to Phase 1 (Foundation).**
