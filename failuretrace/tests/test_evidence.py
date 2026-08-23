"""Evidence-layer tests: retrieval (T12, T13), guidance behavior, summaries, brief."""

from __future__ import annotations

from failuretrace import (
    InterventionContext,
    PromotionRecord,
    brief_for,
    build_fallback,
    build_guidance,
    classify,
    render_brief,
    retrieve_relevant_failures,
    summarize_failures,
)
from failuretrace.evidence import NO_CONTEXT_MESSAGE
from failuretrace.core.enums import CausalSupportLevel, FailureCategory
from failuretrace.core.ids import new_promotion_id
from failuretrace.tests.fixtures.scenarios import (
    inconclusive_noise,
    instability,
    oom_crash,
    overfitting,
)


def _seed(repo, settings, make_trial, ctx, **trial_over):
    """Persist a (trial, fallback-hypothesis) pair for a scenario context."""
    trial = make_trial(**trial_over)
    repo.save_trial(trial)
    classification = classify(ctx, settings)
    hyp = build_fallback(classification, ctx, trial_id=trial.trial_id, settings=settings)
    repo.save_hypothesis(hyp)
    return trial, hyp


# --- T12: relevant records rank above irrelevant --------------------------------
def test_t12_relevant_ranks_above_irrelevant(repo, settings, make_trial):
    _seed(repo, settings, make_trial, instability(),
          changed_components=["optimizer"], hyperparameters={"MATRIX_LR": 0.08})
    _seed(repo, settings, make_trial, overfitting(),
          changed_components=["data"], hyperparameters={"WEIGHT_DECAY": 0.3})

    ic = InterventionContext(
        category=FailureCategory.likely_instability,
        changed_components=["optimizer"],
        changed_hyperparameters={"MATRIX_LR": 0.075},
    )
    results = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    assert results[0].hypothesis.category == FailureCategory.likely_instability
    assert results[0].relevance_score > results[-1].relevance_score


# --- min-score cutoff drops weak / recency-only matches -------------------------
def test_retrieval_min_score_cutoff_drops_weak_matches(make_env, make_trial):
    settings, repo = make_env(ollama_enabled=False, retrieval={"min_relevance_score": 1.0})
    _seed(repo, settings, make_trial, instability(),
          changed_components=["optimizer"], hyperparameters={"MATRIX_LR": 0.08})
    _seed(repo, settings, make_trial, overfitting(),
          git_commit="c2", changed_components=["data"], hyperparameters={"WEIGHT_DECAY": 0.3})
    ic = InterventionContext(
        category=FailureCategory.likely_instability,
        changed_components=["optimizer"],
        changed_hyperparameters={"MATRIX_LR": 0.08},
    )
    results = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    # only the strongly-relevant instability clears the cutoff; the recency-only overfitting
    # match (score well under 1.0) is dropped rather than padding the results.
    assert results
    assert all(rf.hypothesis.category == FailureCategory.likely_instability for rf in results)
    assert all(rf.relevance_score > 1.0 for rf in results)


# --- T13: score explanations present and non-empty ------------------------------
def test_t13_score_explanations_present(repo, settings, make_trial):
    _seed(repo, settings, make_trial, instability(),
          changed_components=["optimizer"], hyperparameters={"MATRIX_LR": 0.08})
    ic = InterventionContext(
        category=FailureCategory.likely_instability,
        changed_components=["optimizer"],
        changed_hyperparameters={"MATRIX_LR": 0.08},
    )
    results = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    assert results
    assert all(rf.score_explanation for rf in results)  # every result explained
    assert results[0].relevance_score > 0


# --- guidance: soft-default, hard only for repeated deterministic / C2+ ----------
def test_guidance_repeated_instability_is_soft_with_warning(repo, settings, make_trial):
    # distinct commits => genuinely distinct trials (same-commit duplicates are deduped)
    for i in range(2):
        _seed(repo, settings, make_trial, instability(),
              trial_id=f"inst{i}", git_commit=f"commit{i}", changed_components=["optimizer"])
    ic = InterventionContext(category=FailureCategory.likely_instability, changed_components=["optimizer"])
    retrieved = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    guidance = build_guidance(retrieved, settings=settings, repository=repo)
    assert guidance.soft_penalties
    assert not guidance.hard_constraints
    assert any("instability" in w for w in guidance.warnings)


def test_guidance_repeated_oom_is_hard_constraint(repo, settings, make_trial):
    for i in range(2):
        _seed(repo, settings, make_trial, oom_crash(),
              trial_id=f"oom{i}", git_commit=f"commit{i}", changed_components=["optimizer"])
    ic = InterventionContext(category=FailureCategory.resource_pressure)
    retrieved = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    guidance = build_guidance(retrieved, settings=settings, repository=repo)
    assert guidance.hard_constraints


def test_guidance_duplicate_commit_does_not_manufacture_hard_constraint(repo, settings, make_trial):
    # The SAME physical OOM recorded twice (same commit) must stay soft — one observation.
    for i in range(2):
        _seed(repo, settings, make_trial, oom_crash(),
              trial_id=f"dup{i}", git_commit="same_commit", changed_components=["optimizer"])
    ic = InterventionContext(category=FailureCategory.resource_pressure)
    retrieved = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    guidance = build_guidance(retrieved, settings=settings, repository=repo)
    assert not guidance.hard_constraints
    assert guidance.soft_penalties


def test_guidance_inconclusive_is_context_only(repo, settings, make_trial):
    _seed(repo, settings, make_trial, inconclusive_noise())
    ic = InterventionContext(category=FailureCategory.inconclusive)
    retrieved = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    guidance = build_guidance(retrieved, settings=settings, repository=repo)
    assert not guidance.hard_constraints
    assert any("inconclusive" in w for w in guidance.warnings)


def test_guidance_c2_evidence_is_hard_constraint(repo, settings, make_trial):
    _, hyp = _seed(repo, settings, make_trial, instability(), changed_components=["optimizer"])
    # supporting trials must be real (write-path gate + FK)
    repo.save_trial(make_trial(trial_id="a", seed=1))
    repo.save_trial(make_trial(trial_id="b", seed=2))
    repo.save_promotion(PromotionRecord(
        promotion_id=new_promotion_id(),
        hypothesis_id=hyp.hypothesis_id,
        from_level=CausalSupportLevel.C1_plausible_hypothesis,
        to_level=CausalSupportLevel.C2_replicated_effect,
        supporting_trial_ids=["a", "b"],
        rationale="two matched-seed replications",
        settings_hash=settings.settings_hash(),
    ))
    ic = InterventionContext(category=FailureCategory.likely_instability, changed_components=["optimizer"])
    retrieved = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    guidance = build_guidance(retrieved, settings=settings, repository=repo)
    assert any("C2" in hc.get("reason", "") for hc in guidance.hard_constraints)


# --- summaries are compact, not raw history -------------------------------------
def test_summaries_are_compact(repo, settings, make_trial):
    _seed(repo, settings, make_trial, instability(), changed_components=["optimizer"])
    ic = InterventionContext(category=FailureCategory.likely_instability, changed_components=["optimizer"])
    retrieved = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    summary = summarize_failures(retrieved)
    assert "likely_instability" in summary
    assert summarize_failures([]) == "No relevant prior failures."


def test_summaries_show_effective_level_after_promotion(repo, settings, make_trial):
    _, hyp = _seed(repo, settings, make_trial, instability(), changed_components=["optimizer"])
    repo.save_trial(make_trial(trial_id="a", seed=1))
    repo.save_trial(make_trial(trial_id="b", seed=2))
    repo.save_promotion(PromotionRecord(
        promotion_id=new_promotion_id(), hypothesis_id=hyp.hypothesis_id,
        from_level=CausalSupportLevel.C1_plausible_hypothesis,
        to_level=CausalSupportLevel.C2_replicated_effect,
        supporting_trial_ids=["a", "b"], rationale="two replications",
        settings_hash=settings.settings_hash(),
    ))
    ic = InterventionContext(category=FailureCategory.likely_instability, changed_components=["optimizer"])
    retrieved = retrieve_relevant_failures(ic, repository=repo, settings=settings)
    # the summary reflects the *effective* (post-promotion) level, not the frozen original
    assert "C2_replicated_effect" in summarize_failures(retrieved)


# --- Phase 8: the pre-experiment brief ------------------------------------------
def _promote_to_c2(repo, settings, make_trial, hypothesis_id):
    repo.save_trial(make_trial(trial_id="sup_a", seed=1))
    repo.save_trial(make_trial(trial_id="sup_b", seed=2))
    repo.save_promotion(PromotionRecord(
        promotion_id=new_promotion_id(), hypothesis_id=hypothesis_id,
        from_level=CausalSupportLevel.C1_plausible_hypothesis,
        to_level=CausalSupportLevel.C2_replicated_effect,
        supporting_trial_ids=["sup_a", "sup_b"], rationale="two replications",
        settings_hash=settings.settings_hash(),
    ))


def test_brief_c1_only_yields_no_binding_constraint(repo, settings, make_trial):
    """A single unreplicated failure is context, never a binding rule."""
    _seed(repo, settings, make_trial, instability(), changed_components=["optimizer"])
    ic = InterventionContext(
        category=FailureCategory.likely_instability, changed_components=["optimizer"]
    )
    brief = brief_for(ic, settings=settings, repository=repo)
    assert not brief.hard_constraints
    assert brief.plausible_context
    assert all("not causally validated" in line for line in brief.plausible_context)


def test_brief_marks_c1_context_as_not_causal_in_markdown(repo, settings, make_trial):
    _seed(repo, settings, make_trial, instability(), changed_components=["optimizer"])
    ic = InterventionContext(
        category=FailureCategory.likely_instability, changed_components=["optimizer"]
    )
    rendered = render_brief(
        brief_for(ic, settings=settings, repository=repo), fmt="markdown", settings=settings
    )
    assert "NOT causally validated" in rendered
    assert "Binding constraints" not in rendered  # nothing has earned one


def test_brief_c2_evidence_becomes_binding_constraint(repo, settings, make_trial):
    _, hyp = _seed(repo, settings, make_trial, instability(), changed_components=["optimizer"])
    _promote_to_c2(repo, settings, make_trial, hyp.hypothesis_id)

    ic = InterventionContext(
        category=FailureCategory.likely_instability, changed_components=["optimizer"]
    )
    brief = brief_for(ic, settings=settings, repository=repo)
    assert brief.hard_constraints
    # promoted evidence is represented once, as a constraint - not also as "plausible"
    assert not brief.plausible_context
    assert "Binding constraints" in render_brief(brief, fmt="markdown", settings=settings)


def test_brief_is_bounded_and_reports_truncation(make_env, make_trial):
    settings, repo = make_env(ollama_enabled=False, brief={"max_items": 2})
    for i, scenario in enumerate((instability(), overfitting(), oom_crash(), inconclusive_noise())):
        _seed(repo, settings, make_trial, scenario,
              trial_id=f"t{i}", git_commit=f"commit{i}", changed_components=["optimizer"])

    ic = InterventionContext(
        category=FailureCategory.likely_instability, changed_components=["optimizer"]
    )
    brief = brief_for(ic, settings=settings, repository=repo, top_k=5)
    assert len(brief.plausible_context) == 2   # four distinct findings, bound of two
    assert brief.truncated >= 2
    assert "omitted by the configured bound" in render_brief(brief, fmt="markdown", settings=settings)


def test_brief_collapses_identical_findings_into_one_line(make_env, make_trial):
    """Repeating one sentence N times would spend the bound without adding information."""
    settings, repo = make_env(ollama_enabled=False)
    for i in range(3):
        _seed(repo, settings, make_trial, instability(),
              trial_id=f"inst{i}", git_commit=f"commit{i}", changed_components=["optimizer"])

    ic = InterventionContext(
        category=FailureCategory.likely_instability, changed_components=["optimizer"]
    )
    brief = brief_for(ic, settings=settings, repository=repo, top_k=5)
    assert len(brief.plausible_context) == 1
    assert "x3 records" in brief.plausible_context[0]
    # a record count is not a replication claim - the finding stays C1
    assert "C1_plausible_hypothesis" in brief.plausible_context[0]
    assert not brief.hard_constraints


def test_brief_respects_max_chars_for_prose_but_not_json(make_env, make_trial):
    settings, repo = make_env(ollama_enabled=False, brief={"max_chars": 120})
    for i in range(4):
        _seed(repo, settings, make_trial, instability(),
              trial_id=f"inst{i}", git_commit=f"commit{i}", changed_components=["optimizer"])
    ic = InterventionContext(
        category=FailureCategory.likely_instability, changed_components=["optimizer"]
    )
    brief = brief_for(ic, settings=settings, repository=repo)

    markdown = render_brief(brief, fmt="markdown", settings=settings)
    assert "truncated at the configured character bound" in markdown
    # JSON must stay parseable, so the char bound does not apply to it
    import json

    assert json.loads(render_brief(brief, fmt="json", settings=settings))["truncated"] >= 0


def test_brief_empty_store_reports_no_context(repo, settings):
    ic = InterventionContext(category=FailureCategory.likely_instability)
    brief = brief_for(ic, settings=settings, repository=repo)
    assert brief.is_empty
    assert render_brief(brief, fmt="markdown", settings=settings).endswith(NO_CONTEXT_MESSAGE)
    assert render_brief(brief, fmt="text", settings=settings) == NO_CONTEXT_MESSAGE
