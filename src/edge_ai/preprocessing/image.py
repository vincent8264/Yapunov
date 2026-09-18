"""Image preprocessing suitable for common vision models."""

import cv2
import numpy as np
from numpy.typing import NDArray


def preprocess_image(
    frame: NDArray[np.uint8],
    size: tuple[int, int] = (224, 224),
    *,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """Resize a BGR frame, convert it to RGB, and optionally scale to [0, 1]."""
    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty image")
    resized = cv2.resize(frame, size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    output = rgb.astype(np.float32)
    if normalize:
        output /= 255.0
    return output
