"""Reporting: matplotlib-optional summaries and failure maps.

Text/markdown artifacts are always produced (pure-Python); PNG plots are produced only
when matplotlib is installed (``plots.py`` degrades to no-ops otherwise). Reports visibly
separate observations / plausible hypotheses / replicated / counterfactual-supported /
robust findings and never present C0/C1 as causal conclusions.
"""

from .failure_map import FailureMapRow, build_failure_map, render_failure_map_text, write_failure_map
from .summary import ReportSummary, build_summary, render_summary_text, write_summary
from .trial import render_trial_text, write_trial_report

__all__ = [
    "ReportSummary",
    "build_summary",
    "render_summary_text",
    "write_summary",
    "FailureMapRow",
    "build_failure_map",
    "render_failure_map_text",
    "write_failure_map",
    "render_trial_text",
    "write_trial_report",
]
