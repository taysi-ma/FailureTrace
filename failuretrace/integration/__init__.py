"""autoresearch integration: the thin, flag-guarded adapter approved in Phase 0.

autoresearch has no callable ratchet (the accept/reject logic lives in ``program.md`` +
``results.tsv`` + git), so integration is an adapter over the artifacts the loop
produces — never a wrapped function, never a change to ``train.py``/``prepare.py``.
See ``docs/failuretrace_integration_report.md``.
"""

from .autoresearch_adapter import (
    ingest_results_tsv,
    record_from_run,
    record_rejected_trial,
    render_program_md_hook,
)
from .optimizer_adapter import guidance_for, soft_penalty_terms

__all__ = [
    "record_rejected_trial",
    "record_from_run",
    "render_program_md_hook",
    "ingest_results_tsv",
    "guidance_for",
    "soft_penalty_terms",
]
