from edge_ai.decision import Decision, decide
from edge_ai.inference.base import InferenceResult


def test_normal_result_is_idle() -> None:
    assert decide(InferenceResult("normal", 0.99)) == Decision("idle")


def test_high_confidence_anomaly_is_alert() -> None:
    assert decide(InferenceResult("anomaly", 0.95)) == Decision("alert")
