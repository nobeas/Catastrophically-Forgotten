"""Placeholder implementation for predictive coding."""

from __future__ import annotations

from src.models.base import MultiLayerPerceptron


class PredictiveCodingMLP(MultiLayerPerceptron):
    """Stub class for a predictive coding variant of the baseline MLP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Predictive coding is not implemented yet. Implement the rule here.")
