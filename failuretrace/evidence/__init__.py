"""Evidence layer: deterministic retrieval, guidance, and compact summaries."""

from .guidance import SearchGuidance, build_guidance
from .retrieval import (
    InterventionContext,
    RetrievedFailure,
    load_retrieval_config,
    retrieve_relevant_failures,
)
from .summaries import (
    NO_CONTEXT_MESSAGE,
    ExperimentBrief,
    brief_for,
    build_brief,
    load_brief_config,
    render_brief,
    summarize_failures,
    summarize_guidance,
)

__all__ = [
    "InterventionContext",
    "RetrievedFailure",
    "retrieve_relevant_failures",
    "load_retrieval_config",
    "SearchGuidance",
    "build_guidance",
    "summarize_failures",
    "summarize_guidance",
    "ExperimentBrief",
    "build_brief",
    "brief_for",
    "render_brief",
    "load_brief_config",
    "NO_CONTEXT_MESSAGE",
]
