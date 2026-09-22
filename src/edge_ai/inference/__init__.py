"""Inference engines and shared result type."""

import os

# Set before importing ONNX Runtime in any engine module. This project intentionally
# keeps inference local and does not opt into platform telemetry.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inference.dummy import DummyInferenceEngine

__all__ = ["DummyInferenceEngine", "InferenceEngine", "InferenceResult"]
