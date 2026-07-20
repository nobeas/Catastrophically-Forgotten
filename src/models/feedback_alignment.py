"""Placeholder implementation for feedback alignment."""

from __future__ import annotations

from src.models.base import MultiLayerPerceptron


class FeedbackAlignmentMLP(MultiLayerPerceptron):
    """Stub class for a feedback alignment variant of the baseline MLP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Feedback alignment is not implemented yet. Implement the rule here.")
