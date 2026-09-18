"""Input sources."""

from edge_ai.inputs.base import InputSource
from edge_ai.inputs.simulated_sensor import SimulatedSensorInput
from edge_ai.inputs.webcam import WebcamInput

__all__ = ["InputSource", "SimulatedSensorInput", "WebcamInput"]
