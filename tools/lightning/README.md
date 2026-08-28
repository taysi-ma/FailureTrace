# Running real autoresearch trials on a free GPU (Lightning AI)

autoresearch's `train.py` hard-requires **FlashAttention-3**, which only runs on **Hopper
(H100/H200)**. This bundle lets you generate **real** trials on a cheaper/free GPU and feed
them into FailureTrace end-to-end.

| File | What it does |
|---|---|
| `patch_train.py` | Swaps the one FA3 call for PyTorch `scaled_dot_product_attention`; precision is unchanged. |
| `patch_t4.py` | Applies the Colab/T4 **FP32** profile: depth 4, sequence 1024, device batch 8, full causal attention, bounded validation. FP32 is deliberate — this trainer assumes bf16's exponent range and ships no GradScaler, so fp16 overflows in the ReLU-squared MLP, the Newton-Schulz iteration, and the fused AdamW. |
| `run_trials.py` | Runs `train.py` with A100 or T4 configs, captures each real `run.log`, ingests into FailureTrace, walks the ladder, and prints levels + effect size. |
| `free_gpu_trials.ipynb` | One-click notebook for **Kaggle / Colab / Lightning**: detects the GPU, picks the profile, applies the right patches, sizes the batch to the card, prepares data, and runs the trials. Start here if you have no GPU of your own. |

## Which GPU
- **Lightning free tier**: 15 credits/month (**monthly**, not daily; ~80 T4-hrs, ~3 A100-hrs).
- **H200/H100** (Hopper): run **unmodified** — skip `patch_train.py`.
- **A100 / L4**: apply `patch_train.py`; bf16 remains enabled.
- **T4 (16 GB)**: apply both patches. T4 has no native bf16 path, and the H100-sized
  sequence/batch defaults do not fit its memory. The profile runs FP32 (no tensor cores,
  ~8.1 vs ~65 TFLOPS), so the fixed 300 s budget buys fewer steps — fine for a pipeline test.

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
# T4 only:
python tools/lightning/patch_t4.py autoresearch
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

For a T4/Colab runtime, use a fresh append-only store and the T4 trial profile:
```bash
python tools/lightning/run_trials.py --profile t4 --repo autoresearch \
    --data-dir ft_data_t4 --reports-dir ft_reports_t4
```
The runner now applies each configured seed to the real trainer, records the actual
autoresearch `HEAD`, fingerprints the effective source/configuration, and aborts before
writing failure evidence when the baseline did not produce `val_bpb`. A counterfactual is
never persisted as `completed` unless its run finished successfully.

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
- **T4 (16 GB):** the profile uses `DEPTH=8`, `MAX_SEQ_LEN=1024`,
  `DEVICE_BATCH_SIZE=8`, `TOTAL_BATCH_SIZE=2**14`, `WINDOW_PATTERN="L"`, and fp32.
  A measured depth-4/seq-512 run peaked at 691.5 MB of 14.56 GiB, so this larger shape
  still leaves wide headroom.
  Run `prepare.py` only after applying `patch_t4.py`, because token batches and validation
  use the patched sequence length.
- **Test the pipeline with no GPU:** `run_trials.py --from-logs <dir>` ingests existing
  `<label>.log` files instead of training — useful to validate FailureTrace wiring locally
  (this is how the bundle was tested).

## What this does NOT touch
`patch_train.py` edits **your** clone's `train.py` only. It does not modify FailureTrace, and
the pinned `./autoresearch` reconnaissance clone in this repo stays byte-for-byte unchanged.
