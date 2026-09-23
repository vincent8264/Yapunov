"""YAMNet-specific ONNX inference and AudioSet-to-project label mapping."""

from collections.abc import Callable, Mapping, Sequence
import csv
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from edge_ai.inference.base import InferenceEngine, InferenceResult

ort.disable_telemetry_events()


# Indices are from Google's YAMNet 521-class map. Groups deliberately contain only
# acoustically relevant labels; the impact event is not represented as a confirmed fall.
DEFAULT_EVENT_CLASS_INDICES: Mapping[str, tuple[int, ...]] = {
    "smoke_alarm": (382, 393, 394),  # Alarm; Smoke detector; Fire alarm
    "glass_break": (435, 437, 464),  # Glass; Shatter; Breaking
    # Basketball bounce is included because genuine project thump recordings rank it
    # strongly; this remains a fall-like impact proxy, not confirmation of a fall.
    "fall_thud": (454, 455, 459, 460, 462, 463),
}

SessionFactory = Callable[..., Any]


def load_yamnet_class_labels(path: Path) -> tuple[str, ...]:
    """Load and validate YAMNet's fixed 521-row AudioSet class map."""
    try:
        with Path(path).open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"YAMNet class map not found: {path}") from exc
    except csv.Error as exc:
        raise ValueError(f"could not read YAMNet class map {path}: {exc}") from exc

    if len(rows) != 521:
        raise ValueError("YAMNet class map must contain exactly 521 labels")
    labels: list[str] = []
    for expected_index, row in enumerate(rows):
        try:
            index = int(row["index"])
            label = row["display_name"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("YAMNet class map must contain index and display_name columns") from exc
        if index != expected_index or not label:
            raise ValueError("YAMNet class map indices must run from 0 through 520")
        labels.append(label)
    return tuple(labels)


class YAMNetInferenceEngine(InferenceEngine):
    """Run waveform-input YAMNet and reduce its multi-label scores to project events."""

    CLASS_COUNT = 521

    def __init__(
        self,
        model_path: Path,
        *,
        background_threshold: float = 0.1,
        event_class_indices: Mapping[str, Sequence[int]] = DEFAULT_EVENT_CLASS_INDICES,
        class_map_path: Path | None = None,
        class_labels: Sequence[str] | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"YAMNet ONNX model not found: {path}")
        if not 0.0 <= background_threshold <= 1.0:
            raise ValueError("YAMNet background_threshold must be between 0 and 1")
        if not event_class_indices:
            raise ValueError("YAMNet event class mapping must not be empty")

        normalized_mapping: dict[str, tuple[int, ...]] = {}
        for event, indices in event_class_indices.items():
            normalized = tuple(indices)
            if not event or not normalized:
                raise ValueError("YAMNet event class mapping contains an empty event")
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self.CLASS_COUNT
                for index in normalized
            ):
                raise ValueError(
                    f"YAMNet event {event!r} contains a class index outside 0..520"
                )
            normalized_mapping[event] = normalized

        factory = session_factory or ort.InferenceSession
        self._session = factory(str(path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 1:
            raise ValueError("YAMNet model must have one rank-1 waveform input")
        self.input_name = inputs[0].name

        outputs = self._session.get_outputs()
        scores_outputs = [
            output
            for output in outputs
            if output.name == "output_0"
            or (len(output.shape) == 2 and output.shape[-1] == self.CLASS_COUNT)
        ]
        if len(scores_outputs) != 1:
            raise ValueError("YAMNet model must expose one [frames, 521] scores output")
        self.scores_output_name = scores_outputs[0].name
        self.background_threshold = background_threshold
        self.event_class_indices = normalized_mapping
        if class_labels is None:
            class_labels = load_yamnet_class_labels(
                class_map_path or path.with_name("yamnet_class_map.csv")
            )
        if len(class_labels) != self.CLASS_COUNT or any(not label for label in class_labels):
            raise ValueError("YAMNet class labels must contain exactly 521 non-empty labels")
        self.class_labels = tuple(class_labels)

    def predict(self, data: Any) -> InferenceResult:
        scores = self._predict_scores(data)

        raw_index = int(np.argmax(scores))
        raw_frame, raw_class_index = np.unravel_index(raw_index, scores.shape)
        raw_confidence = float(np.clip(scores[raw_frame, raw_class_index], 0.0, 1.0))
        raw_label = self.class_labels[raw_class_index]

        event_scores = {
            event: float(np.max(scores[:, indices]))
            for event, indices in self.event_class_indices.items()
        }
        event, confidence = max(event_scores.items(), key=lambda item: item[1])
        confidence = float(np.clip(confidence, 0.0, 1.0))
        common = {
            "model_label": raw_label,
            "model_confidence": raw_confidence,
            "source": "yamnet",
            "evaluated_events": tuple(self.event_class_indices),
        }
        if confidence < self.background_threshold:
            return InferenceResult("background", 1.0 - confidence, **common)
        return InferenceResult(event, confidence, **common)

    def top_classes(self, data: Any, *, limit: int = 5) -> tuple[tuple[str, float], ...]:
        """Return the strongest raw AudioSet classes, aggregated over model frames."""
        if isinstance(limit, bool) or not 1 <= limit <= self.CLASS_COUNT:
            raise ValueError("YAMNet top-class limit must be between 1 and 521")
        scores = self._predict_scores(data)
        aggregated = np.max(scores, axis=0)
        indices = np.argsort(aggregated)[::-1][:limit]
        return tuple(
            (self.class_labels[int(index)], float(np.clip(aggregated[index], 0.0, 1.0)))
            for index in indices
        )

    def _predict_scores(self, data: Any) -> np.ndarray:
        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("YAMNet input must be a non-empty rank-1 waveform")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("YAMNet input waveform must contain only finite values")

        outputs = self._session.run(
            [self.scores_output_name],
            {self.input_name: waveform},
        )
        if len(outputs) != 1:
            raise ValueError("YAMNet did not return its scores output")
        scores = np.asarray(outputs[0], dtype=np.float32)
        if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] != self.CLASS_COUNT:
            raise ValueError(
                f"YAMNet scores must have shape [frames, 521], got {scores.shape}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("YAMNet scores must contain only finite values")
        return scores
