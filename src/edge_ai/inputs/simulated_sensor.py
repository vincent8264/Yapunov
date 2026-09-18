"""Repeatable artificial sensor input for laptop development."""

import random

from edge_ai.inputs.base import InputSource


class SimulatedSensorInput(InputSource):
    def __init__(
        self,
        normal_value: float = 0.3,
        noise: float = 0.03,
        anomaly_probability: float = 0.1,
        anomaly_value: float = 1.3,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= anomaly_probability <= 1.0:
            raise ValueError("anomaly_probability must be between 0 and 1")
        if noise < 0.0:
            raise ValueError("noise must be non-negative")
        self.normal_value = normal_value
        self.noise = noise
        self.anomaly_probability = anomaly_probability
        self.anomaly_value = anomaly_value
        self.last_value: float | None = None
        self._random = random.Random(seed)

    def read(self) -> float:
        center = (
            self.anomaly_value
            if self._random.random() < self.anomaly_probability
            else self.normal_value
        )
        self.last_value = self._random.gauss(center, self.noise)
        return self.last_value
