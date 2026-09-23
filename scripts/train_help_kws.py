"""Train and export a small ``help`` classifier on frozen YAMNet embeddings.

The exported ONNX graph contains both YAMNet's waveform frontend and the learned
classifier, so the application still receives one raw 16 kHz waveform and returns
one probability.  Model and dataset files are intentionally git-ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from edge_ai.inputs.audio import read_wav
from edge_ai.preprocessing.audio import prepare_audio_waveform, resample_audio_frame


@dataclass(frozen=True)
class Classifier:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float


def _paths(dataset: Path, split: str) -> tuple[list[Path], np.ndarray]:
    paths: list[Path] = []
    labels: list[float] = []
    for folder, label in (("background", 0.0), ("help_call", 1.0)):
        found = sorted((dataset / split / folder).rglob("*.wav"))
        if not found:
            raise ValueError(f"no WAV files found in {dataset / split / folder}")
        paths.extend(found)
        labels.extend([label] * len(found))
    return paths, np.asarray(labels, dtype=np.float32)


def _extend_class(
    paths: list[Path],
    labels: np.ndarray,
    directory: Path | None,
    *,
    label: float,
) -> tuple[list[Path], np.ndarray]:
    if directory is None:
        return paths, labels
    extra = sorted(directory.rglob("*.wav"))
    if not extra:
        raise ValueError(f"no additional WAV files found in {directory}")
    return paths + extra, np.concatenate(
        (labels, np.full(len(extra), label, dtype=np.float32))
    )


def _feature_for_waveform(waveform: np.ndarray, session: object) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    embeddings = np.asarray(
        session.run(["output_1"], {input_name: waveform.astype(np.float32)})[0],
        dtype=np.float32,
    )
    return np.concatenate((embeddings.mean(axis=0), embeddings.max(axis=0)))


def _features(paths: list[Path], session: object) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index, path in enumerate(paths, start=1):
        waveform = prepare_audio_waveform(read_wav(path))
        rows.append(_feature_for_waveform(waveform, session))
        if index % 100 == 0 or index == len(paths):
            print(f"embedded {index}/{len(paths)} clips", flush=True)
    return np.stack(rows)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _classifier_scores(features: np.ndarray, classifier: Classifier) -> np.ndarray:
    normalized = (features - classifier.mean) / classifier.scale
    return _sigmoid(normalized @ classifier.weights + classifier.bias)


def _max_clip_scores(
    paths: list[Path], session: object, classifier: Classifier
) -> np.ndarray:
    """Score every overlapping one-second window and retain each clip maximum."""
    scores: list[float] = []
    window_size = 16_000
    step_size = 3_200
    for index, path in enumerate(paths, start=1):
        samples = resample_audio_frame(read_wav(path), sample_rate=16_000)
        starts = list(range(0, max(1, samples.size - window_size + 1), step_size))
        final_start = max(0, samples.size - window_size)
        if starts[-1] != final_start:
            starts.append(final_start)
        features: list[np.ndarray] = []
        for start in starts:
            waveform = samples[start : start + window_size]
            if waveform.size < window_size:
                waveform = np.pad(waveform, (0, window_size - waveform.size))
            features.append(_feature_for_waveform(waveform, session))
        scores.append(float(np.max(_classifier_scores(np.stack(features), classifier))))
        if index % 20 == 0 or index == len(paths):
            print(f"scored {index}/{len(paths)} long validation clips", flush=True)
    return np.asarray(scores, dtype=np.float32)


def _hard_negative_features(
    paths: list[Path], session: object, classifier: Classifier, *, per_clip: int = 2
) -> np.ndarray:
    """Mine the most help-like windows from long background recordings."""
    selected: list[np.ndarray] = []
    window_size = 16_000
    step_size = 3_200
    for index, path in enumerate(paths, start=1):
        samples = resample_audio_frame(read_wav(path), sample_rate=16_000)
        starts = list(range(0, max(1, samples.size - window_size + 1), step_size))
        final_start = max(0, samples.size - window_size)
        if starts[-1] != final_start:
            starts.append(final_start)
        features: list[np.ndarray] = []
        for start in starts:
            waveform = samples[start : start + window_size]
            if waveform.size < window_size:
                waveform = np.pad(waveform, (0, window_size - waveform.size))
            features.append(_feature_for_waveform(waveform, session))
        feature_array = np.stack(features)
        scores = _classifier_scores(feature_array, classifier)
        for hard_index in np.argsort(scores)[-per_clip:]:
            selected.append(feature_array[hard_index])
        if index % 50 == 0 or index == len(paths):
            print(f"mined {index}/{len(paths)} background clips", flush=True)
    return np.stack(selected)


def train_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    steps: int = 2_000,
    learning_rate: float = 0.03,
    l2: float = 0.001,
) -> Classifier:
    """Fit standardized logistic regression with deterministic full-batch Adam."""
    mean = train_x.mean(axis=0).astype(np.float32)
    scale = train_x.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-5))
    x = ((train_x - mean) / scale).astype(np.float32)
    validation = ((validation_x - mean) / scale).astype(np.float32)

    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = np.float32(0.0)
    weight_m = np.zeros_like(weights)
    weight_v = np.zeros_like(weights)
    bias_m = np.float32(0.0)
    bias_v = np.float32(0.0)
    positive_count = max(1, int(np.sum(train_y == 1.0)))
    negative_count = max(1, int(np.sum(train_y == 0.0)))
    sample_weights = np.where(
        train_y == 1.0,
        x.shape[0] / (2.0 * positive_count),
        x.shape[0] / (2.0 * negative_count),
    )
    for step in range(1, steps + 1):
        probabilities = _sigmoid(x @ weights + bias)
        error = (probabilities - train_y) * sample_weights
        weight_gradient = x.T @ error / x.shape[0] + l2 * weights
        bias_gradient = np.float32(error.mean())
        weight_m = 0.9 * weight_m + 0.1 * weight_gradient
        weight_v = 0.999 * weight_v + 0.001 * np.square(weight_gradient)
        bias_m = np.float32(0.9 * bias_m + 0.1 * bias_gradient)
        bias_v = np.float32(0.999 * bias_v + 0.001 * bias_gradient**2)
        correction_m = 1.0 - 0.9**step
        correction_v = 1.0 - 0.999**step
        weights -= learning_rate * (weight_m / correction_m) / (
            np.sqrt(weight_v / correction_v) + 1e-8
        )
        bias -= learning_rate * (bias_m / correction_m) / (
            np.sqrt(bias_v / correction_v) + 1e-8
        )

    scores = _sigmoid(validation @ weights + bias)
    threshold = select_threshold(validation_y, scores)
    return Classifier(mean, scale, weights, float(bias), threshold)


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select the threshold with the highest F1, then balanced accuracy."""
    best: tuple[float, float, float] | None = None
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = scores >= threshold
        positive = labels == 1.0
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & ~positive))
        tn = int(np.sum(~predicted & ~positive))
        fn = int(np.sum(~predicted & positive))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        candidate = (f1, (recall + specificity) / 2.0, float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[2]


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> str:
    predicted = scores >= threshold
    positive = labels == 1.0
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / labels.size
    return (
        f"TP={tp} FP={fp} TN={tn} FN={fn} precision={precision:.3f} "
        f"recall={recall:.3f} accuracy={accuracy:.3f}"
    )


def export_model(yamnet: Path, classifier: Classifier, output: Path) -> None:
    """Append the classifier to YAMNet's embedding output and replace its outputs."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(yamnet)
    graph = model.graph
    graph.node.extend(
        [
            helper.make_node(
                "ReduceMean", ["output_1"], ["help_embedding_mean"], axes=[0], keepdims=0
            ),
            helper.make_node(
                "ReduceMax", ["output_1"], ["help_embedding_max"], axes=[0], keepdims=0
            ),
            helper.make_node(
                "Concat",
                ["help_embedding_mean", "help_embedding_max"],
                ["help_embedding"],
                axis=0,
            ),
            helper.make_node(
                "Sub", ["help_embedding", "help_feature_mean"], ["help_centered"]
            ),
            helper.make_node(
                "Div", ["help_centered", "help_feature_scale"], ["help_normalized"]
            ),
            helper.make_node(
                "Mul", ["help_normalized", "help_classifier_weights"], ["help_weighted"]
            ),
            helper.make_node(
                "ReduceSum",
                ["help_weighted", "help_reduce_axis"],
                ["help_logit_no_bias"],
                keepdims=1,
            ),
            helper.make_node(
                "Add", ["help_logit_no_bias", "help_classifier_bias"], ["help_logit"]
            ),
            helper.make_node("Sigmoid", ["help_logit"], ["help_probability"]),
        ]
    )
    graph.initializer.extend(
        [
            numpy_helper.from_array(classifier.mean.astype(np.float32), "help_feature_mean"),
            numpy_helper.from_array(classifier.scale.astype(np.float32), "help_feature_scale"),
            numpy_helper.from_array(
                classifier.weights.astype(np.float32), "help_classifier_weights"
            ),
            numpy_helper.from_array(
                np.asarray([classifier.bias], dtype=np.float32), "help_classifier_bias"
            ),
            numpy_helper.from_array(
                np.asarray([0], dtype=np.int64), "help_reduce_axis"
            ),
        ]
    )
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info("help_probability", TensorProto.FLOAT, [1])
    )
    helper.set_model_props(
        model,
        {
            "edge_ai.keyword": "help",
            "edge_ai.frontend": "YAMNet output_1 mean+max embeddings",
            "edge_ai.activation_threshold": f"{classifier.threshold:.3f}",
        },
    )
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--additional-train-background", type=Path)
    parser.add_argument("--additional-train-help", type=Path)
    parser.add_argument("--additional-validation-background", type=Path)
    parser.add_argument("--yamnet", type=Path, default=Path("models/yamnet.onnx"))
    parser.add_argument("--output", type=Path, default=Path("models/help-kws.onnx"))
    args = parser.parse_args()

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.yamnet), providers=["CPUExecutionProvider"])
    train_paths, train_y = _paths(args.dataset, "train")
    validation_paths, validation_y = _paths(args.dataset, "validation")
    hard_negative_paths: list[Path] = []
    if args.additional_train_background is not None:
        hard_negative_paths = sorted(args.additional_train_background.rglob("*.wav"))
    train_paths, train_y = _extend_class(
        train_paths, train_y, args.additional_train_background, label=0.0
    )
    train_paths, train_y = _extend_class(
        train_paths, train_y, args.additional_train_help, label=1.0
    )
    train_x = _features(train_paths, session)
    validation_x = _features(validation_paths, session)
    classifier = train_classifier(train_x, train_y, validation_x, validation_y)
    if hard_negative_paths:
        hard_negative_x = _hard_negative_features(
            hard_negative_paths, session, classifier
        )
        train_x = np.concatenate((train_x, hard_negative_x))
        train_y = np.concatenate(
            (train_y, np.zeros(hard_negative_x.shape[0], dtype=np.float32))
        )
        classifier = train_classifier(
            train_x, train_y, validation_x, validation_y
        )
    validation_scores = _classifier_scores(validation_x, classifier)
    threshold_labels = validation_y
    threshold_scores = validation_scores
    if args.additional_validation_background is not None:
        real_validation = sorted(args.additional_validation_background.rglob("*.wav"))
        if not real_validation:
            raise ValueError(
                "no additional validation background WAV files found in "
                f"{args.additional_validation_background}"
            )
        real_scores = _max_clip_scores(real_validation, session, classifier)
        threshold_labels = np.concatenate(
            (validation_y, np.zeros(real_scores.size, dtype=np.float32))
        )
        threshold_scores = np.concatenate((validation_scores, real_scores))
        classifier = replace(
            classifier,
            threshold=select_threshold(threshold_labels, threshold_scores),
        )
    print(f"selected threshold={classifier.threshold:.3f}")
    print(
        "held-out validation:",
        metrics(threshold_labels, threshold_scores, classifier.threshold),
    )
    export_model(args.yamnet, classifier, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
