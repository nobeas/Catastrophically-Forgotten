"""Placeholder implementation for Hebbian learning."""

from __future__ import annotations

from src.models.base import MultiLayerPerceptron


class HebbianMultiLayerPerceptron(MultiLayerPerceptron):
    """Stub class for a Hebbian variant of the baseline MLP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Hebbian learning is not implemented yet. Implement the rule here.")
