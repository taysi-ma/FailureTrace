#!/usr/bin/env python3
"""Run the FailureTrace end-to-end demo: ``python demo/run_demo.py``.

Equivalent to ``python -m failuretrace demo``. Ollama is disabled throughout; everything
runs CPU-only and offline against the synthetic fixture set.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failuretrace.demo import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
