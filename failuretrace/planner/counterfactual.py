"""Deterministic counterfactual planner. Returns plans only — never executes anything.

Generates one controlled validation experiment for a selected category, holding every
known variable that is not intervened on constant. Expected outcomes are stated
direction-aware (interpreted via ``improvement()`` at gate time).
"""

from __future__ import annotations

import logging

from ..core.models import CounterfactualPlan, FailureHypothesis
from ..core.ids import new_plan_id
from ..core.settings import Settings
from .interventions import DEFAULT_KNOWN_VARIABLES, FAMILIES

logger = logging.getLogger(__name__)


def plan_counterfactual(
    hypothesis: FailureHypothesis,
    *,
    settings: Settings,
    known_variables: list[str] | None = None,
) -> CounterfactualPlan | None:
    """Build a controlled counterfactual plan, or ``None`` if the category has no family."""
    family = FAMILIES.get(hypothesis.category)
    if family is None:
        logger.debug("no counterfactual family for category %s", hypothesis.category)
        return None

    known = list(known_variables) if known_variables is not None else list(DEFAULT_KNOWN_VARIABLES)
    treatment = [family.primary_variable]
    if family.coupled_variable is not None:
        treatment.append(family.coupled_variable)
    treatment_set = set(treatment)

    # Everything known that is not intervened on is held constant (plus coarse holds).
    held_constant = sorted((set(known) - treatment_set) | set(family.also_hold))
    control = [f"{variable}@baseline" for variable in treatment]

    direction = settings.metric.direction.value
    expected_true = (
        f"treatment yields a positive direction-aware improvement (improvement > 0 under "
        f"{direction}) versus the held-constant control"
    )
    expected_false = (
        f"no directional improvement (improvement <= 0 under {direction}); "
        f"the hypothesis is not supported"
    )

    plan = CounterfactualPlan(
        plan_id=new_plan_id(),
        hypothesis_id=hypothesis.hypothesis_id,
        primary_intervention_variable=family.primary_variable,
        optional_coupled_stabilization_variable=family.coupled_variable,
        control_variables=control,
        treatment_variables=treatment,
        held_constant_variables=held_constant,
        expected_outcome_if_hypothesis_true=expected_true,
        expected_outcome_if_hypothesis_false=expected_false,
        interaction_rationale=family.interaction_rationale,
        settings_hash=settings.settings_hash(),
    )
    logger.debug("planned counterfactual %s for hypothesis %s", plan.plan_id, hypothesis.hypothesis_id)
    return plan
