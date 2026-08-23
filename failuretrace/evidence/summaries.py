"""Compact, prompt-friendly summaries. Never returns the raw full history.

Also hosts the **pre-experiment brief** (Phase 8): the retrieval half of the loop,
rendered for an agent that is about to choose its next experiment. Two properties are
load-bearing:

- **Bounded.** Only the top-ranked evidence is injected and the remainder is reported as
  a count. Long contexts are used poorly in the middle (Liu et al., TACL 2024), so
  dumping raw history back into the agent would defeat the purpose.
- **Epistemically honest.** Binding constraints (C2+ / deterministic repeated resource
  failure) are separated from advisory penalties and from C0/C1 context, which is
  labeled as not causally validated — the same separation the reports use. The brief is
  consumed by a model that will act on it, so an overstated claim here becomes a real
  experimental decision.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..core.enums import CausalSupportLevel
from ..core.settings import Settings
from .guidance import SearchGuidance, build_guidance
from .retrieval import InterventionContext, RetrievedFailure, retrieve_relevant_failures


def summarize_failures(retrieved: list[RetrievedFailure], *, max_items: int = 5) -> str:
    if not retrieved:
        return "No relevant prior failures."
    lines = []
    for rf in retrieved[:max_items]:
        hyp = rf.hypothesis
        headline = hyp.hypotheses[0] if hyp.hypotheses else hyp.category.value
        # Prefer the effective (post-promotion) level so a promoted hypothesis is not
        # understated; fall back to the record's own level when not supplied.
        level = (rf.effective_level or hyp.causal_support_level).value
        lines.append(
            f"- [{hyp.category.value}] {headline} (score={rf.relevance_score:.2f}, {level})"
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


# --- pre-experiment brief -------------------------------------------------------
NO_CONTEXT_MESSAGE = "No relevant prior failures for this experiment."

_TRUNCATION_MARKER = "\n… brief truncated at the configured character bound."


class BriefConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_items: int = 5
    max_chars: int = 2000


def load_brief_config(settings: Settings) -> BriefConfig:
    return BriefConfig(**settings.section("brief"))


class ExperimentBrief(BaseModel):
    """Bounded, honestly-sectioned negative evidence for one candidate experiment."""

    model_config = ConfigDict(extra="forbid")

    hard_constraints: list[str] = Field(default_factory=list)
    soft_penalties: list[str] = Field(default_factory=list)
    plausible_context: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # How many items the configured bound dropped, across all sections. Reported rather
    # than silently discarded so the agent knows the brief is a subset.
    truncated: int = 0

    @property
    def is_empty(self) -> bool:
        return not (
            self.hard_constraints or self.soft_penalties
            or self.plausible_context or self.warnings
        )


def _bound(items: list[str], limit: int) -> tuple[list[str], int]:
    if limit <= 0 or len(items) <= limit:
        return items, 0
    return items[:limit], len(items) - limit


def _entry_line(entry: dict) -> str:
    subject = entry.get("variable") or entry.get("category") or "unspecified"
    reason = entry.get("reason") or ""
    return f"{subject}: {reason}".strip().rstrip(":")


def build_brief(
    retrieved: list[RetrievedFailure],
    guidance: SearchGuidance,
    *,
    settings: Settings,
) -> ExperimentBrief:
    """Assemble the bounded brief from retrieved failures and their derived guidance.

    Constraint entries come from ``guidance`` (which already enforces that a hard
    constraint requires C2+ effective support or a repeated deterministic resource
    failure). ``plausible_context`` carries only C0/C1 items and never asserts a cause.
    """
    cfg = load_brief_config(settings)

    hard = [_entry_line(e) for e in guidance.hard_constraints]
    soft = [_entry_line(e) for e in guidance.soft_penalties]

    # Identical findings are collapsed: N rule-based hypotheses of the same category
    # produce the same sentence, and repeating it N times would burn the bound without
    # adding information. The count is reported as "xN records" — deliberately a statement
    # about how many records match, NOT a replication claim. Whether repetition amounts to
    # independent evidence is decided by the guidance layer, which de-duplicates by source
    # commit; a promoted hypothesis appears above as a binding constraint instead.
    grouped: dict[tuple[str, str, str], dict] = {}
    for rf in retrieved:
        level = rf.effective_level or rf.hypothesis.causal_support_level
        if level.at_least(CausalSupportLevel.C2_replicated_effect):
            continue  # already represented as a binding constraint
        hyp = rf.hypothesis
        headline = hyp.hypotheses[0] if hyp.hypotheses else hyp.category.value
        key = (hyp.category.value, headline, level.value)
        entry = grouped.setdefault(key, {"score": rf.relevance_score, "count": 0})
        entry["count"] += 1
        entry["score"] = max(entry["score"], rf.relevance_score)

    context: list[str] = []
    for (category, headline, level_value), entry in grouped.items():
        repeats = f", x{entry['count']} records" if entry["count"] > 1 else ""
        context.append(
            f"[{category}] {headline} "
            f"(score={entry['score']:.2f}, {level_value}{repeats}, not causally validated)"
        )

    hard, dropped_hard = _bound(hard, cfg.max_items)
    soft, dropped_soft = _bound(soft, cfg.max_items)
    context, dropped_context = _bound(context, cfg.max_items)
    warnings, dropped_warnings = _bound(list(guidance.warnings), cfg.max_items)

    return ExperimentBrief(
        hard_constraints=hard,
        soft_penalties=soft,
        plausible_context=context,
        warnings=warnings,
        truncated=dropped_hard + dropped_soft + dropped_context + dropped_warnings,
    )


def brief_for(
    intervention_context: InterventionContext,
    *,
    settings: Settings,
    repository,
    top_k: int = 5,
) -> ExperimentBrief:
    """Retrieve, derive guidance, and assemble the brief for a candidate experiment."""
    retrieved = retrieve_relevant_failures(
        intervention_context, repository=repository, settings=settings, top_k=top_k
    )
    guidance = build_guidance(retrieved, settings=settings, repository=repository)
    return build_brief(retrieved, guidance, settings=settings)


def _render_markdown(brief: ExperimentBrief) -> str:
    lines = ["## FailureTrace brief — prior negative evidence", ""]
    if brief.is_empty:
        lines.append(NO_CONTEXT_MESSAGE)
        return "\n".join(lines)

    if brief.hard_constraints:
        lines.append("### Binding constraints (C2+ or repeated deterministic failure)")
        lines += [f"- {item}" for item in brief.hard_constraints]
        lines.append("")
    if brief.soft_penalties:
        lines.append("### Advisory (soft penalties — prefer to avoid, not forbidden)")
        lines += [f"- {item}" for item in brief.soft_penalties]
        lines.append("")
    if brief.plausible_context:
        lines.append("### Plausible hypotheses (C0/C1 — NOT causally validated)")
        lines += [f"- {item}" for item in brief.plausible_context]
        lines.append("")
    if brief.warnings:
        lines.append("### Context")
        lines += [f"- {item}" for item in brief.warnings]
        lines.append("")
    if brief.truncated:
        lines.append(f"_{brief.truncated} further item(s) omitted by the configured bound._")
    return "\n".join(lines).rstrip()


def _render_text(brief: ExperimentBrief) -> str:
    if brief.is_empty:
        return NO_CONTEXT_MESSAGE
    lines: list[str] = []
    for label, items in (
        ("HARD (binding)", brief.hard_constraints),
        ("soft (advisory)", brief.soft_penalties),
        ("plausible (C0/C1, NOT causally validated)", brief.plausible_context),
        ("context", brief.warnings),
    ):
        for item in items:
            lines.append(f"[{label}] {item}")
    if brief.truncated:
        lines.append(f"... {brief.truncated} further item(s) omitted by the configured bound.")
    return "\n".join(lines)


def render_brief(brief: ExperimentBrief, *, fmt: str = "markdown", settings: Settings | None = None) -> str:
    """Render the brief as ``markdown``, ``text``, or ``json``.

    ``max_chars`` bounds the prose formats only; truncating JSON would produce an
    invalid document, and it is already bounded by ``max_items``.
    """
    if fmt == "json":
        return brief.model_dump_json(indent=2)

    rendered = _render_markdown(brief) if fmt == "markdown" else _render_text(brief)
    max_chars = load_brief_config(settings).max_chars if settings is not None else 0
    if max_chars > 0 and len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + _TRUNCATION_MARKER
    return rendered
