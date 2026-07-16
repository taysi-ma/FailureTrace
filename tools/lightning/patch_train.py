#!/usr/bin/env python3
"""Patch autoresearch's ``train.py``: FlashAttention-3 (Hopper-only) -> torch SDPA.

autoresearch imports a FlashAttention-3 kernel that only runs on Hopper (H100/H200).
This swaps the single FA3 call for PyTorch's ``scaled_dot_product_attention`` so the same
``train.py`` runs on **any** CUDA GPU (A100, L4, T4, ...). bf16 is kept (native on A100).
The sliding-window ("S") layers are preserved via an additive causal-band mask, so the
model is numerically the same as with FA3 (SDPA uses FA2 under the hood on A100).

Idempotent. Only edits the FA3 import block. Usage:

    python patch_train.py /path/to/autoresearch/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# The exact FA3 block shipped in autoresearch train.py (pinned 228791f).
_OLD = '''from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface'''

_NEW = '''# --- FailureTrace/Lightning patch: FlashAttention-3 -> torch SDPA ---------------
# Runs on any CUDA GPU (A100/L4/T4) instead of requiring Hopper FA3. bf16 kept.
# Sliding-window ("S") layers preserved via an additive causal-band mask.
import types as _ft_types

def _ft_sdpa_flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
    # FA layout (B, T, H, D) -> SDPA layout (B, H, T, D)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    T = q.size(2)
    left = window_size[0] if window_size is not None else -1
    if left is None or left < 0 or left >= T:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=bool(causal))
    else:
        idx = torch.arange(T, device=q.device)
        diff = idx[:, None] - idx[None, :]                 # query i - key j
        keep = (diff >= 0) & (diff <= int(left))           # causal window [i-left, i]
        attn_mask = torch.zeros(T, T, dtype=q.dtype, device=q.device).masked_fill(~keep, float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
    return y.transpose(1, 2).contiguous()                  # back to (B, T, H, D)

fa3 = _ft_types.SimpleNamespace(flash_attn_func=_ft_sdpa_flash_attn_func)
# --- end patch -----------------------------------------------------------------'''


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "train.py")
    src = path.read_text(encoding="utf-8")
    if "_ft_sdpa_flash_attn_func" in src:
        print(f"{path}: already patched — nothing to do")
        return 0
    if _OLD not in src:
        print(
            f"ERROR: the expected FlashAttention-3 block was not found in {path}.\n"
            "This patch targets autoresearch pinned at 228791f; adapt _OLD if your copy differs.",
            file=sys.stderr,
        )
        return 1
    path.write_text(src.replace(_OLD, _NEW), encoding="utf-8")
    print(f"{path}: patched FlashAttention-3 -> torch SDPA (now runs on A100/L4/T4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
