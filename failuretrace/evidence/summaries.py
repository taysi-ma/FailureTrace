"""Compact, prompt-friendly summaries. Never returns the raw full history."""

from __future__ import annotations

from .guidance import SearchGuidance
from .retrieval import RetrievedFailure


def summarize_failures(retrieved: list[RetrievedFailure], *, max_items: int = 5) -> str:
    if not retrieved:
        return "No relevant prior failures."
    lines = []
    for rf in retrieved[:max_items]:
        hyp = rf.hypothesis
        headline = hyp.hypotheses[0] if hyp.hypotheses else hyp.category.value
        lines.append(
            f"- [{hyp.category.value}] {headline} "
            f"(score={rf.relevance_score:.2f}, {hyp.causal_support_level.value})"
        )
    if len(retrieved) > max_items:
        lines.append(f"- ... and {len(retrieved) - max_items} more")
    return "\n".join(lines)


def summarize_guidance(guidance: SearchGuidance) -> str:
    parts = []
    if guidance.hard_constraints:
        parts.append(f"{len(guidance.hard_constraints)} hard constraint(s)")
    if guidance.soft_penalties:
        parts.append(f"{len(guidance.soft_penalties)} soft penalty(ies)")
    if guidance.warnings:
        parts.append(f"{len(guidance.warnings)} warning(s)")
    return "; ".join(parts) if parts else "no guidance"
