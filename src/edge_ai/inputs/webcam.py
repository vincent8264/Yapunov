"""OpenCV webcam input."""

from typing import Any

import cv2

from edge_ai.inputs.base import InputSource


class WebcamInput(InputSource):
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open webcam at camera index {camera_index}")

    def read(self) -> Any:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read a frame from camera index {self.camera_index}")
        return frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "WebcamInput":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
