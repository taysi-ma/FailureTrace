#!/usr/bin/env python3
"""Apply a conservative FP32/T4 profile to a pinned autoresearch clone.

The upstream defaults target an H100: depth 8, sequence length 2048, device batch 128,
and bf16. A 16 GB Tesla T4 cannot execute that profile. This idempotent patch halves the
model depth and selects a substantially smaller runtime shape while leaving the fixed
five-minute experiment budget unchanged.

Precision is FP32, not FP16. Turing has no bf16, but a naive bf16 -> fp16 swap does not
work either: this trainer assumes bf16's exponent range throughout and ships no
GradScaler. fp16 overflows at 65504 in at least three places -- the ReLU-squared MLP
(train.py:108, overflows once any activation exceeds 256), the Polar Express /
Newton-Schulz iteration's A @ A term (train.py:324), and grad.square() inside the fused
AdamW step (train.py:310). Observed on a Kaggle T4: NaN loss and FAIL after three
optimizer steps. FP32 loses the tensor cores (~8.1 vs ~65 TFLOPS), so the fixed 300 s
budget simply buys fewer steps -- which is irrelevant for a pipeline test that only needs
a valid val_bpb.

Run ``patch_train.py`` first to replace FlashAttention-3 with torch SDPA, then run:

    python tools/lightning/patch_t4.py autoresearch
"""

from __future__ import annotations

import sys
from pathlib import Path


_TRAIN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
        'WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context',
        'WINDOW_PATTERN = "L"    # FailureTrace T4 profile: memory-efficient causal SDPA',
    ),
    (
        "TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step",
        "TOTAL_BATCH_SIZE = 2**14 # FailureTrace T4 profile: 16K tokens per optimizer step",
    ),
    (
        "DEPTH = 8               # number of transformer layers",
        "DEPTH = 4               # FailureTrace T4 profile: half-depth model",
    ),
    (
        "DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)",
        "DEVICE_BATCH_SIZE = 8    # FailureTrace T4 profile: fits 16 GB VRAM",
    ),
    ("self.transformer.wte.to(dtype=torch.bfloat16)", "self.transformer.wte.to(dtype=torch.float32)"),
    ("ve.to(dtype=torch.bfloat16)", "ve.to(dtype=torch.float32)"),
    ("cos, sin = cos.bfloat16(), sin.bfloat16()", "cos, sin = cos.float(), sin.float()"),
    ("X = g.bfloat16()", "X = g.float()"),
    (
        'autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)',
        'autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)',
    ),
    ("H100_BF16_PEAK_FLOPS = 989.5e12", "ACCELERATOR_PEAK_FLOPS = 8.1e12"),
    ("/ H100_BF16_PEAK_FLOPS", "/ ACCELERATOR_PEAK_FLOPS"),
)

_PREPARE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
        "MAX_SEQ_LEN = 2048       # context length",
        "MAX_SEQ_LEN = 512        # FailureTrace T4 profile: reduced context length",
    ),
    (
        "EVAL_TOKENS = 40 * 524288  # number of tokens for val eval",
        "EVAL_TOKENS = 2 * 524288   # FailureTrace T4 profile: bounded validation cost",
    ),
)


def _apply_replacements(path: Path, replacements: tuple[tuple[str, str], ...]) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            raise ValueError(f"expected pinned autoresearch text not found in {path}: {old!r}")
        text = text.replace(old, new)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_repo(repo: Path) -> bool:
    """Patch ``train.py`` and ``prepare.py`` under *repo*; return whether either changed."""
    train_changed = _apply_replacements(repo / "train.py", _TRAIN_REPLACEMENTS)
    prepare_changed = _apply_replacements(repo / "prepare.py", _PREPARE_REPLACEMENTS)
    return train_changed or prepare_changed


def main() -> int:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else "autoresearch")
    try:
        changed = patch_repo(repo)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    state = "applied" if changed else "already applied"
    print(f"{repo}: T4 FP32 profile {state} (depth=4, seq=512, device_batch=8).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
