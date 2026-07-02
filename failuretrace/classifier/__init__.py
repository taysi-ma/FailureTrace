"""Deterministic, explainable, threshold-driven failure classifier."""

from .classifier import ClassificationContext, FailureClassification, classify
from .thresholds import ConfidenceTiers, Thresholds, load_confidence, load_thresholds

__all__ = [
    "ClassificationContext",
    "FailureClassification",
    "classify",
    "Thresholds",
    "ConfidenceTiers",
    "load_thresholds",
    "load_confidence",
]
