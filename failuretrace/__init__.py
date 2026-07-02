"""FailureTrace — provider-free, failure-aware experiment governance for autoresearch.

Phase 1 public surface: core enums, Pydantic models, Settings + the canonical
``improvement()`` helper, ID generators, idempotent database initialization, and the
repository (the only write path). ``record_rejected_trial()`` (the autoresearch-facing
public API) lands in Phase 5.
"""

from __future__ import annotations

from .core.enums import (
    CausalSupportLevel,
    FailureCategory,
    HypothesisSource,
    LinkType,
    MetricDirection,
    TrialStatus,
)
from .core.ids import (
    new_hypothesis_id,
    new_link_id,
    new_plan_id,
    new_promotion_id,
    new_replication_group_id,
    new_trial_id,
)
from .core.models import (
    CounterfactualPlan,
    CounterfactualPlanRef,
    FailureHypothesis,
    Intervention,
    LinkRecord,
    PromotionRecord,
    TrialRecord,
)
from .core.settings import Settings, get_settings, improvement, load_settings
from .store.errors import (
    DuplicateRecordError,
    HardConstraintViolation,
    PromotionViolation,
    ReferentialIntegrityError,
    StoreError,
)
from .store.migrations import initialize_database
from .store.repository import Repository
from .telemetry import TelemetryRecord, normalize, parse_run_log, telemetry_from_run_log
from .classifier import ClassificationContext, FailureClassification, classify
from .analyst import OllamaClient, OllamaError, analyze, build_fallback
from .evidence import (
    InterventionContext,
    RetrievedFailure,
    SearchGuidance,
    build_guidance,
    retrieve_relevant_failures,
    summarize_failures,
)
from .planner import (
    CounterfactualResult,
    ReplicationEvidence,
    evaluate_c4,
    evaluate_counterfactual,
    evaluate_replication,
    plan_counterfactual,
    promote_replications,
)
from .integration import (
    guidance_for,
    record_from_run,
    record_rejected_trial,
    render_program_md_hook,
)
from .demo import DemoResult, run_demo

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # enums
    "TrialStatus",
    "FailureCategory",
    "CausalSupportLevel",
    "HypothesisSource",
    "MetricDirection",
    "LinkType",
    # models
    "TrialRecord",
    "FailureHypothesis",
    "Intervention",
    "CounterfactualPlanRef",
    "CounterfactualPlan",
    "PromotionRecord",
    "LinkRecord",
    # settings + helper
    "Settings",
    "load_settings",
    "get_settings",
    "improvement",
    # ids
    "new_trial_id",
    "new_hypothesis_id",
    "new_promotion_id",
    "new_plan_id",
    "new_link_id",
    "new_replication_group_id",
    # store
    "initialize_database",
    "Repository",
    "StoreError",
    "DuplicateRecordError",
    "HardConstraintViolation",
    "ReferentialIntegrityError",
    "PromotionViolation",
    # telemetry
    "TelemetryRecord",
    "normalize",
    "parse_run_log",
    "telemetry_from_run_log",
    # classifier
    "ClassificationContext",
    "FailureClassification",
    "classify",
    # analyst
    "analyze",
    "build_fallback",
    "OllamaClient",
    "OllamaError",
    # evidence
    "InterventionContext",
    "RetrievedFailure",
    "retrieve_relevant_failures",
    "SearchGuidance",
    "build_guidance",
    "summarize_failures",
    # planner
    "plan_counterfactual",
    "ReplicationEvidence",
    "CounterfactualResult",
    "evaluate_replication",
    "evaluate_counterfactual",
    "evaluate_c4",
    "promote_replications",
    # integration (autoresearch-facing public API)
    "record_rejected_trial",
    "record_from_run",
    "render_program_md_hook",
    "guidance_for",
    # demo
    "run_demo",
    "DemoResult",
]
