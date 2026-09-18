"""Simple sensor normalization and optional fixed-size windows."""

from collections import deque

import numpy as np
from numpy.typing import NDArray


def normalize_sensor(value: float, mean: float = 0.0, scale: float = 1.0) -> float:
    if scale == 0.0:
        raise ValueError("scale must not be zero")
    return (float(value) - mean) / scale


class SlidingWindow:
    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("size must be at least 1")
        self._values: deque[float] = deque(maxlen=size)

    def __call__(self, value: float) -> NDArray[np.float32]:
        self._values.append(float(value))
        return np.asarray(self._values, dtype=np.float32)
