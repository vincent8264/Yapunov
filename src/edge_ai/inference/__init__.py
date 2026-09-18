"""Inference engines and shared result type."""

from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inference.dummy import DummyInferenceEngine

__all__ = ["DummyInferenceEngine", "InferenceEngine", "InferenceResult"]
