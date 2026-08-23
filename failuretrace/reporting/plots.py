"""Matplotlib plotting helpers — optional. No-ops (return ``None``) if matplotlib is
absent, so the rest of reporting works with no extra dependencies. Uses the headless
``Agg`` backend; never imports torch or touches a GPU.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

# Set once pyplot has been proven unimportable in this process. A failed import is not
# cached by Python, so without this every plot call would repeat the work — and that work
# includes building the font cache, which on some systems takes minutes and can fail
# (e.g. matplotlib's macOS font query raises KeyError when system_profiler returns an
# unexpected plist). Degrading once is the intent; degrading once per figure is a hang.
_pyplot_unavailable = False


def _pyplot():
    """Return the ``Agg``-backed pyplot module, or ``None`` if it cannot be imported."""
    global _pyplot_unavailable
    if _pyplot_unavailable:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - an unusable matplotlib is not an error here
        _pyplot_unavailable = True
        logger.warning(
            "matplotlib is installed but unusable (%s: %s); plots will be skipped",
            exc.__class__.__name__, exc,
        )
        return None
    return plt


def bar_plot(counts: dict[str, float], title: str, path: str | Path) -> Path | None:
    """Save a labeled bar chart of ``counts``; ``None`` if matplotlib absent / no data."""
    if not HAS_MATPLOTLIB or not counts:
        return None
    plt = _pyplot()
    if plt is None:
        return None
    try:
        keys = list(counts)
        values = [counts[k] for k in keys]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(keys)), values, color="#c0504d")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return Path(path)
    except Exception as exc:  # noqa: BLE001 - plotting must never break a report
        logger.warning("bar_plot failed (%s): %s", exc.__class__.__name__, exc)
        return None


def scatter_plot(
    points: list[tuple[float, float, str]],
    title: str,
    path: str | Path,
    *,
    xlabel: str = "",
    ylabel: str = "",
) -> Path | None:
    """Save a scatter of ``(x, y, group)`` points colored by group; ``None`` if unavailable."""
    if not HAS_MATPLOTLIB or not points:
        return None
    plt = _pyplot()
    if plt is None:
        return None
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        groups = sorted({label for _, _, label in points})
        cmap = plt.get_cmap("tab10")
        for idx, group in enumerate(groups):
            xs = [x for x, _, label in points if label == group]
            ys = [y for _, y, label in points if label == group]
            ax.scatter(xs, ys, label=group, color=cmap(idx % 10), s=40)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if groups:
            ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return Path(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scatter_plot failed (%s): %s", exc.__class__.__name__, exc)
        return None
