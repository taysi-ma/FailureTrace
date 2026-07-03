"""Analyst tests including Gate-3 checks T6-T9. No test requires a running Ollama."""

from __future__ import annotations

import json

import pytest

from failuretrace.analyst import OllamaClient, OllamaError, analyze, build_fallback, build_prompt
from failuretrace.analyst.ollama_client import OllamaConfig
from failuretrace.classifier import classify
from failuretrace.core.enums import CausalSupportLevel, FailureCategory, HypothesisSource
from failuretrace.tests.fixtures.scenarios import (
    SCENARIOS,
    inconclusive_noise,
    instability,
    missing_telemetry,
    oom_crash,
)

# A well-formed LLM response that ALSO tries to smuggle in C3 + a hard constraint
# (both must be ignored by the deterministic-authority merge).
LLM_JSON = json.dumps(
    {
        "observations": ["LLM: gradient variance elevated"],
        "evidence": ["LLM: cv=3.0"],
        "hypotheses": ["LLM: learning rate too high"],
        "alternative_explanations": ["LLM: seed noise", "LLM: data ordering"],
        "missing_evidence": ["LLM: no LR history"],
        "hypothesis_confidence": 0.6,
        "evidence_quality": 0.5,
        "suggested_intervention": {
            "variable": "optimizer.lr", "action": "decrease",
            "target_value": 0.02, "rationale": "halve LR",
        },
        "proposed_counterfactual_trial": {"summary": "halve LR, hold everything else"},
        "category": "possible_overfitting",                       # must be ignored
        "causal_support_level": "C3_counterfactual_supported",  # must be ignored
        "should_apply_hard_constraint": True,                     # must be ignored
    }
)

LLM_CATEGORY_JSON = json.dumps(
    {
        "category": "likely_undertraining",
        "observations": ["LLM: loss was still decreasing near the budget cutoff"],
        "evidence": ["LLM: no deterministic crash signal; compare as a plausible budget issue"],
        "hypotheses": ["LLM: the change may need more training budget to show validation gain"],
        "alternative_explanations": ["LLM: seed noise", "LLM: neutral change"],
        "missing_evidence": ["LLM: no per-step validation curve"],
        "hypothesis_confidence": 0.55,
        "evidence_quality": 0.4,
        "suggested_intervention": {
            "variable": "schedule.horizon", "action": "increase",
            "target_value": None, "rationale": "test whether more budget improves the metric",
        },
        "proposed_counterfactual_trial": {
            "summary": "hold architecture and optimizer fixed; increase training budget"
        },
    }
)


class _StubClient:
    def __init__(self, *, response_text=None, raise_exc=None):
        self.response_text = response_text
        self.raise_exc = raise_exc

    def generate(self, prompt):
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response_text


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, *, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = 0

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._response


# --- fallback alone satisfies the whole pipeline --------------------------------
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_fallback_produces_valid_hypothesis(settings, name):
    factory, expected = SCENARIOS[name]
    ctx = factory()
    classification = classify(ctx, settings)
    hyp = build_fallback(classification, ctx, trial_id="trial_x", settings=settings)
    assert hyp.category == expected
    assert hyp.source == HypothesisSource.rule_based
    assert hyp.causal_support_level.rank <= 1                 # single trial => C0/C1
    assert hyp.hypotheses and hyp.alternative_explanations    # non-empty narrative
    assert 0.0 <= hyp.hypothesis_confidence <= 1.0


# --- T6: Ollama unavailable / garbage -> safe fallback persisted -----------------
def test_t6_ollama_unroutable_falls_back(make_env, make_trial):
    settings, repo = make_env(ollama_enabled=True)
    repo.save_trial(make_trial(trial_id="t6a"))
    ctx = instability()
    classification = classify(ctx, settings)
    client = _StubClient(raise_exc=OllamaError("unroutable URL"))
    hyp = analyze(classification, ctx, trial_id="t6a", settings=settings, client=client, repository=repo)
    assert hyp.source == HypothesisSource.rule_based_fallback
    assert repo.get_hypothesis(hyp.hypothesis_id) is not None


def test_t6_ollama_garbage_json_falls_back(make_env, make_trial):
    settings, repo = make_env(ollama_enabled=True)
    repo.save_trial(make_trial(trial_id="t6b"))
    ctx = instability()
    classification = classify(ctx, settings)
    client = _StubClient(response_text="not valid json {{{")
    hyp = analyze(classification, ctx, trial_id="t6b", settings=settings, client=client, repository=repo)
    assert hyp.source == HypothesisSource.rule_based_fallback
    assert repo.get_hypothesis(hyp.hypothesis_id) is not None


def test_ollama_client_unroutable_raises():
    # Real requests path against a closed localhost port -> OllamaError (no Ollama needed).
    client = OllamaClient(OllamaConfig(base_url="http://127.0.0.1:9", max_retries=0, timeout_seconds=1.0))
    with pytest.raises(OllamaError):
        client.generate("hello")


# --- LLM success enriches, but deterministic fields stay authoritative -----------
def test_llm_success_enriches_hypothesis(make_env, make_trial):
    settings, repo = make_env(ollama_enabled=True)
    repo.save_trial(make_trial(trial_id="t_llm"))
    ctx = instability()
    classification = classify(ctx, settings)
    client = _StubClient(response_text=LLM_JSON)
    hyp = analyze(classification, ctx, trial_id="t_llm", settings=settings, client=client, repository=repo)
    assert hyp.source == HypothesisSource.local_llm
    assert "LLM: gradient variance elevated" in hyp.observations
    assert hyp.category == classification.category  # category stays deterministic
    # the deterministic rubric confidence is authoritative; the LLM's stated 0.6 is only
    # recorded for provenance in llm_confidence, never in hypothesis_confidence (Invariant 5)
    assert hyp.hypothesis_confidence == classification.confidence
    assert hyp.llm_confidence == 0.6


def test_llm_can_refine_unknown_category(make_env, make_trial):
    settings, repo = make_env(ollama_enabled=True)
    repo.save_trial(make_trial(trial_id="t_llm_category"))
    ctx = missing_telemetry()
    classification = classify(ctx, settings)
    assert classification.category == FailureCategory.inconclusive
    client = _StubClient(response_text=LLM_CATEGORY_JSON)
    hyp = analyze(
        classification, ctx,
        trial_id="t_llm_category", settings=settings, client=client, repository=repo,
    )
    assert hyp.source == HypothesisSource.local_llm
    assert hyp.category == FailureCategory.likely_undertraining
    assert hyp.causal_support_level == CausalSupportLevel.C1_plausible_hypothesis
    assert hyp.should_apply_soft_penalty is True
    assert hyp.should_apply_hard_constraint is False
    assert hyp.hypothesis_confidence == classification.confidence
    assert hyp.llm_confidence == 0.55


# --- T7: single-trial hypothesis capped at C1 even if the LLM returns C3 ---------
def test_t7_single_trial_capped_at_c1(make_env):
    settings, _ = make_env(ollama_enabled=True)
    ctx = instability()
    classification = classify(ctx, settings)
    client = _StubClient(response_text=LLM_JSON)  # tries C3 + hard constraint
    hyp = analyze(classification, ctx, trial_id="t7", settings=settings, client=client)
    assert hyp.source == HypothesisSource.local_llm
    assert hyp.causal_support_level.rank <= CausalSupportLevel.C1_plausible_hypothesis.rank
    assert hyp.should_apply_hard_constraint is False


# --- T8: inconclusive evidence never yields a hard constraint --------------------
def test_t8_inconclusive_never_hard(settings):
    ctx = inconclusive_noise()
    classification = classify(ctx, settings)
    hyp = build_fallback(classification, ctx, trial_id="t8", settings=settings)
    assert hyp.category == FailureCategory.inconclusive
    assert hyp.should_apply_hard_constraint is False


# --- T9: single OOM -> no persistent hard constraint unless objective limit exceeded
def test_t9_single_oom_no_hard_constraint_by_default(make_env, make_trial):
    settings, repo = make_env(ollama_enabled=False)  # resource_vram_limit_gb is null
    repo.save_trial(make_trial(trial_id="t9a"))
    ctx = oom_crash()
    classification = classify(ctx, settings)
    hyp = analyze(classification, ctx, trial_id="t9a", settings=settings, repository=repo)
    assert classification.category == FailureCategory.resource_pressure
    assert hyp.should_apply_hard_constraint is False
    assert repo.get_hypothesis(hyp.hypothesis_id).should_apply_hard_constraint is False


def test_t9_single_oom_hard_constraint_when_limit_exceeded(make_env, make_trial):
    settings, repo = make_env(ollama_enabled=False, thresholds={"resource_vram_limit_gb": 40.0})
    repo.save_trial(make_trial(trial_id="t9b"))
    ctx = oom_crash()  # default telemetry peak_vram_gb = 80.0 >= 40.0
    classification = classify(ctx, settings)
    hyp = analyze(classification, ctx, trial_id="t9b", settings=settings, repository=repo)
    assert hyp.should_apply_hard_constraint is True
    assert repo.get_hypothesis(hyp.hypothesis_id).should_apply_hard_constraint is True


# --- prompt content -------------------------------------------------------------
def test_prompt_contains_rules_and_schema(settings):
    ctx = instability()
    classification = classify(ctx, settings)
    prompt = build_prompt(classification, ctx, code_diff_summary="raised LR", changed_components=["optimizer"])
    assert "Return ONLY a JSON object" in prompt
    assert "Never assign causal support above C1" in prompt
    assert "FailureHypothesis" in prompt   # embedded JSON schema title
    assert "raised LR" in prompt


# --- OllamaClient unit behavior (fake session; no network) ----------------------
def test_ollama_client_parses_response_field():
    session = _FakeSession(response=_FakeResponse(200, {"response": '{"observations":["x"]}'}))
    client = OllamaClient(OllamaConfig(), session=session)
    assert client.generate("p") == '{"observations":["x"]}'
    assert session.calls == 1


def test_ollama_client_retries_then_raises():
    session = _FakeSession(exc=ConnectionError("refused"))
    client = OllamaClient(OllamaConfig(max_retries=1), session=session)
    with pytest.raises(OllamaError):
        client.generate("p")
    assert session.calls == 2  # initial attempt + 1 retry


def test_ollama_client_missing_response_field_raises():
    session = _FakeSession(response=_FakeResponse(200, {}))
    client = OllamaClient(OllamaConfig(max_retries=0), session=session)
    with pytest.raises(OllamaError):
        client.generate("p")
