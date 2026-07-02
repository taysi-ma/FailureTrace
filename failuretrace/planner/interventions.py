"""Per-category intervention families for counterfactual planning.

Variable names are the real autoresearch knobs (train.py:432-451, per the Phase-0 report)
where applicable. Exactly one primary variable by default; ``resource_pressure`` uses a
coupled stabilization variable (grad accumulation) because it explicitly concerns the
batch x effective-batch interaction, so it carries an ``interaction_rationale``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.enums import FailureCategory

# Real autoresearch tunable knobs (train.py:432-451) — the variable universe held constant
# unless intervened on. "code_diff"/"architecture"/"optimizer"/"effective_batch_size" are
# coarse groupings the planner may additionally hold fixed.
DEFAULT_KNOWN_VARIABLES: list[str] = [
    "ASPECT_RATIO",
    "HEAD_DIM",
    "WINDOW_PATTERN",
    "TOTAL_BATCH_SIZE",
    "EMBEDDING_LR",
    "UNEMBEDDING_LR",
    "MATRIX_LR",
    "SCALAR_LR",
    "WEIGHT_DECAY",
    "ADAM_BETAS",
    "WARMUP_RATIO",
    "WARMDOWN_RATIO",
    "FINAL_LR_FRAC",
    "DEPTH",
    "DEVICE_BATCH_SIZE",
]


@dataclass(frozen=True)
class InterventionFamily:
    category: FailureCategory
    primary_variable: str
    primary_action: str
    coupled_variable: str | None = None
    coupled_action: str | None = None
    interaction_rationale: str | None = None
    also_hold: tuple[str, ...] = ()
    rationale: str = ""


FAMILIES: dict[FailureCategory, InterventionFamily] = {
    FailureCategory.likely_instability: InterventionFamily(
        category=FailureCategory.likely_instability,
        primary_variable="MATRIX_LR",
        primary_action="decrease",
        also_hold=("code_diff", "effective_batch_size"),
        rationale="reduce the primary learning rate to test optimization instability",
    ),
    FailureCategory.likely_undertraining: InterventionFamily(
        category=FailureCategory.likely_undertraining,
        primary_variable="schedule.horizon",
        primary_action="increase",
        also_hold=("architecture", "optimizer"),
        rationale="increase the training budget / schedule horizon, holding architecture + optimizer",
    ),
    FailureCategory.possible_overfitting: InterventionFamily(
        category=FailureCategory.possible_overfitting,
        primary_variable="WEIGHT_DECAY",
        primary_action="increase",
        also_hold=("architecture",),
        rationale="adjust exactly one regularization variable, holding architecture",
    ),
    FailureCategory.resource_pressure: InterventionFamily(
        category=FailureCategory.resource_pressure,
        primary_variable="DEVICE_BATCH_SIZE",
        primary_action="decrease",
        coupled_variable="grad_accum_steps",
        coupled_action="increase",
        interaction_rationale=(
            "raise gradient accumulation to hold the effective batch size constant while "
            "reducing the per-device batch (memory decreases, effective batch fixed); the plan "
            "tests the device-batch x gradient-accumulation interaction"
        ),
        also_hold=("effective_batch_size", "code_diff"),
        rationale="reduce per-device batch, preserving effective batch via gradient accumulation",
    ),
}
