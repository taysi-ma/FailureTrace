"""Prompt construction for the local LLM hypothesis generator.

The prompt provides normalized telemetry, code-diff summary, changed components, the
hyperparameter delta, the deterministic classifier output, baseline/post metrics, metric
direction, and runtime diagnostics — and instructs the model to behave epistemically and
return ONLY JSON matching the FailureHypothesis schema.
"""

from __future__ import annotations

import json

from ..classifier.classifier import FailureClassification
from ..classifier.context import ClassificationContext
from ..core.models import FailureHypothesis

_INSTRUCTIONS = [
    "You are a careful ML experiment failure analyst examining ONE rejected/failed trial.",
    "Follow these rules strictly:",
    "- Do NOT claim causality from a single experiment.",
    "- Separate observations (facts) from hypotheses (proposed explanations).",
    "- State alternative explanations and what evidence is missing.",
    "- Propose exactly ONE falsifiable counterfactual experiment.",
    "- Use ONLY the evidence provided below; never invent telemetry, code behavior, or results.",
    "- Never assign causal support above C1 for a single trial.",
    "- Never recommend a hard constraint unless the evidence shows deterministic repeated failure.",
    "- Return ONLY a JSON object matching the schema; output no prose outside the JSON.",
]


def _hyperparameter_delta(ctx: ClassificationContext) -> dict[str, dict]:
    delta: dict[str, dict] = {}
    for key, new_value in ctx.changed_hyperparameters.items():
        delta[key] = {"from": ctx.baseline_hyperparameters.get(key), "to": new_value}
    return delta


def build_prompt(
    classification: FailureClassification,
    ctx: ClassificationContext,
    *,
    code_diff_summary: str | None = None,
    changed_components: list[str] | None = None,
) -> str:
    schema = json.dumps(FailureHypothesis.model_json_schema(), indent=2)
    telemetry = ctx.telemetry.model_dump(exclude_none=True)

    lines: list[str] = []
    lines.extend(_INSTRUCTIONS)
    lines.append("")
    lines.append("=== Deterministic classifier result (authoritative for category) ===")
    lines.append(f"category: {classification.category.value}")
    lines.append(f"confidence: {classification.confidence}")
    lines.append(f"triggered_rules: {classification.triggered_rules}")
    lines.append(
        f"alternative_categories: {[c.value for c in classification.alternative_categories]}"
    )
    lines.append(f"observations: {classification.observations}")
    lines.append("")
    lines.append("=== Trial evidence ===")
    lines.append(
        f"metric_direction: {ctx.metric_direction.value}; "
        f"baseline: {ctx.baseline_metric}; post: {ctx.post_change_metric}"
    )
    lines.append(
        f"runtime_diagnostics: exception_type={ctx.exception_type}, "
        f"exception_message={ctx.exception_message}, finished={ctx.finished}"
    )
    lines.append(f"normalized_telemetry: {json.dumps(telemetry)}")
    lines.append(f"hyperparameter_delta: {json.dumps(_hyperparameter_delta(ctx))}")
    lines.append(f"changed_components: {changed_components or []}")
    lines.append(f"code_diff_summary: {code_diff_summary or '(not provided)'}")
    lines.append("")
    lines.append("=== Respond with ONLY JSON matching this FailureHypothesis schema ===")
    lines.append(schema)
    return "\n".join(lines)
