# CLAUDE.md — FailureTrace Project Working Rules

You are building **failuretrace**, a provider-free, failure-aware experiment governance
layer around the existing autoresearch repository. The full specification is in
`FAILURETRACE_SPEC.md`. Read it before doing anything.

## Phase discipline (non-negotiable)

- Work is divided into Phases 0–6 as defined in the spec. **Never work ahead of the
  current phase.** Complete the phase, run its verification gate, report results, and
  STOP until the user approves the next phase.
- Every phase ends with: (1) `pytest` output shown in full, (2) a short summary of files
  added/modified, (3) any deviations from the spec with justification, (4) an explicit
  "awaiting approval for Phase N+1" statement.
- Commit at the end of every phase with message `failuretrace: phase N — <summary>`.
  Never commit failing tests. Never amend or rewrite prior phase commits.
- If a phase's tests fail and you cannot fix them within the phase's scope, stop and
  report — do not weaken tests to make them pass, do not skip tests, do not mark xfail
  without user approval.

## Ground rules

- **Provider-free.** Never add dependencies on OpenAI, Anthropic, Gemini, W&B cloud,
  paid vector DBs, paid APIs, or hosted inference. Ollama is the only permitted LLM
  endpoint and it must be optional. If you find yourself wanting a cloud service, stop
  and ask.
- **Do not modify autoresearch's core behavior.** The only permitted modifications to
  the existing repo are the minimal integration hooks defined in Phase 5, guarded by a
  feature flag, plus optional telemetry emission if strictly required. When
  `failuretrace.enabled: false`, autoresearch must behave byte-for-byte identically.
- **Epistemic honesty is a hard requirement.** A single rejected trial NEVER produces
  causal support above C1. Hard constraints require deterministic repeated failure, an
  objectively exceeded configured resource limit, or C2+ support. Never present C0/C1
  findings as causal conclusions in code, reports, or your own summaries.
- **Deterministic first.** Prefer rule-based logic over LLM inference whenever
  deterministic evidence is available. All tests must pass CPU-only with Ollama absent.
- **No invented interfaces.** Integration points must reference real files, real
  functions, and real line-level anchors found by inspecting the actual repo (Phase 0).
  If a clean insertion point does not exist, write a thin adapter — never pretend an
  interface exists.
- **No scope creep.** Do not build: multi-agent systems, code-editing agents, web
  dashboards, custom Bayesian optimizers, a causal inference framework, embedding
  pipelines, or an Optuna sampler. If it isn't in the spec, don't build it.

## Code standards

- Python type hints everywhere. Pydantic models for every persisted or external schema.
- Structured logging (`logging` with module-level loggers; no prints in library code).
- Small, testable functions. No hidden global state. No hard-coded machine paths —
  all data/report paths flow through the Settings object.
- Persistence is append-only. Records are immutable after write; changes are expressed
  as new linked records.
- Minimal dependencies: pydantic, pyyaml, pandas, matplotlib, pytest, requests (for the
  Ollama client). Nothing else without asking. No LangChain. Optuna only if a tested
  adapter or demo actually uses it.
- Every threshold, weight, and limit lives in `failuretrace/config/defaults.yaml` and is
  loaded through Settings. No magic numbers in classifier, retrieval, or gate logic.
