# Running real autoresearch trials on a free GPU (Lightning AI)

autoresearch's `train.py` hard-requires **FlashAttention-3**, which only runs on **Hopper
(H100/H200)**. This bundle lets you generate **real** trials on a cheaper/free GPU and feed
them into FailureTrace end-to-end.

| File | What it does |
|---|---|
| `patch_train.py` | Swaps the one FA3 call for PyTorch `scaled_dot_product_attention` → `train.py` runs on **any** CUDA GPU (A100/L4/T4). bf16 kept; sliding windows preserved (numerically verified). |
| `run_trials.py` | Runs `train.py` with a few tunable configs, captures each real `run.log`, ingests into FailureTrace, walks the ladder, prints levels + effect size. |

## Which GPU
- **Lightning free tier**: 15 credits/month (**monthly**, not daily; ~80 T4-hrs, ~3 A100-hrs).
- **H200/H100** (Hopper): run **unmodified** — skip `patch_train.py`.
- **A100 / L4 / T4**: apply `patch_train.py` first (bf16 stays on A100/L4; on T4 also shrink the model — see below).

## Steps (cost-optimized: data prep on free CPU, GPU only for training)

**1 — On a FREE CPU Studio (burns no GPU credit):**
```bash
git clone <autoresearch-repo> autoresearch
cd autoresearch && uv sync
uv run prepare.py --num-shards 8        # download data + train tokenizer (CPU is fine)
cd .. && pip install -e <path-to-failuretrace>   # or: pip install failuretrace
```

**2 — Apply the patch (A100/L4/T4 only; skip on H200):**
```bash
python tools/lightning/patch_train.py autoresearch/train.py
```

**3 — Switch the Studio to an A100, then run the trials (~30–45 min):**
```bash
python tools/lightning/run_trials.py --repo autoresearch \
    --data-dir ft_data --reports-dir ft_reports
```
It runs a baseline + two oversized-batch runs (→ OOM ×2 → `resource_pressure` → C2) + one
reduced-batch "fix" (the controlled counterfactual → C3), then prints:
```
=== governance summary (real evidence) ===
  resource_pressure  C3_counterfactual_supported  effect=+0.14 (n=1)
report: ft_reports/summary.md
```

**4 — Inspect** `ft_reports/summary.md`, or copy `ft_data/` back to your laptop and run
`python -m failuretrace report summary` / `effects` there. Everything after training is CPU-only.

## Notes & customization
- **Real outcomes vary.** The category/level reflects genuine dynamics. The default configs
  target the `resource_pressure` path because it is the deterministic, *plannable* route to a
  real C3. Other failures (divergence via NaN, plain regressions) classify honestly but may
  stay C0/C1 (autoresearch's `run.log` carries no gradient telemetry, so instability rules
  can't fire).
- **Want a confidence interval?** Add a second `counterfactual` config (another reduced-batch
  run) — an effect interval needs n ≥ 2 counterfactuals.
- **Edit the experiment:** change `CONFIGS` in `run_trials.py` (or pass `--configs my.json`).
  Each entry sets `train.py` constants (`DEVICE_BATCH_SIZE`, `MATRIX_LR`, `DEPTH`, ...).
- **T4 (16 GB):** also shrink per autoresearch's README — `DEPTH=4`, lower `MAX_SEQ_LEN`
  (in `prepare.py`), `TOTAL_BATCH_SIZE=2**14`, `WINDOW_PATTERN="L"`; bf16 works but fp16 is
  faster on Turing.
- **Test the pipeline with no GPU:** `run_trials.py --from-logs <dir>` ingests existing
  `<label>.log` files instead of training — useful to validate FailureTrace wiring locally
  (this is how the bundle was tested).

## What this does NOT touch
`patch_train.py` edits **your** clone's `train.py` only. It does not modify FailureTrace, and
the pinned `./autoresearch` reconnaissance clone in this repo stays byte-for-byte unchanged.
