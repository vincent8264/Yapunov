# Models

Place hackathon ONNX models in this directory. Model binaries are git-ignored.

For every selected model, record:

- filename, source URL, version/hash, and license;
- input name, shape, layout, dtype, color order, resize/crop behavior, and normalization;
- label order;
- output names/shapes and whether outputs are logits, probabilities, embeddings, or
  another representation;
- the adapter and decision threshold used by this project;
- representative-device latency and any conversion or quantization steps.

Do not set `output_type = "probabilities"` merely to make configuration load. Confirm
the model actually includes the required sigmoid/softmax activation. For logits or
task-specific outputs, add and test a model-specific `output_adapter`.

## Selected sound model: YAMNet ONNX

- Source: `audiomagic/yamnet-onnx`, revision `f25b741`, a straight ONNX conversion of
  Google's `google/yamnet/1` release:
  <https://huggingface.co/audiomagic/yamnet-onnx>
- License: Apache-2.0 for the model and code; the included AudioSet ontology labels
  are CC BY 4.0.
- Local filename: `yamnet.onnx` (model binaries remain git-ignored).
- Verified SHA-256:
  `d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e`.
- Input: one dynamic-length, rank-1 `float32` waveform named `waveform`; mono PCM at
  16 kHz, nominal range `[-1, 1]`. Do not peak-normalize live input.
- Internal preprocessing: log-mel frontend is included in the ONNX graph. Each model
  frame covers 0.96 seconds and advances 0.48 seconds.
- Outputs: `output_0` is `[frames, 521]` sigmoid AudioSet class scores; `output_1` is
  `[frames, 1024]` embeddings; `output_2` is the 64-band log-mel representation.
- Label order: the fixed 521-row `yamnet_class_map.csv` distributed with the model,
  committed here so runtime output can name the model's top AudioSet class. Its
  SHA-256 is `cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2`.

Download the model once during setup:

```bash
curl -L https://huggingface.co/audiomagic/yamnet-onnx/resolve/main/yamnet.onnx \
  -o models/yamnet.onnx
```

`YAMNetInferenceEngine` takes the maximum score over model frames and these AudioSet
groups:

- `smoke_alarm`: Alarm (382), Smoke detector/smoke alarm (393), Fire alarm (394)
- `glass_break`: Glass (435), Shatter (437), Breaking (464)
- `fall_thud`: Thump/thud (454), Thunk (455), Basketball bounce (459), Bang (460),
  Whack/thwack (462), Smash/crash (463). Basketball bounce is an empirically useful
  impact proxy for the project's recorded thumps; actual ball bounces are therefore
  an expected false-positive risk.

`fall_thud` means a heavy impact signature, not confirmation that a person fell.
Scores are multi-label probabilities rather than a softmax distribution. The live
configuration's thresholds are unvalidated starting points, not safety claims.

The runner prints both `label` (the project event or `background`, which drives the
alert policy) and `model_label`/`model_confidence` (the highest raw model result).
The latter is diagnostic only and never triggers hardware or notifications by itself.

Before the field demo, validate genuine MOVO USB-M1 recordings at different ranges,
ordinary household noise, television playback, dishes, doors, dropped objects, and
other confusing negatives. Record per-class precision/recall, a confusion matrix,
representative UNO Q latency, and the final threshold values here.

The repository's `spectral_demo` engine is only a deterministic integration baseline
for simulated audio. It must not be used to claim real-world detection accuracy.

## Laptop transcription model: Sherpa Zipformer English 66M

This is a laptop-only speech-to-text evaluation. The standalone `transcribe` command
does not feed its transcript into the alert policy. The separate laptop-only
`help-asr-live.example.toml` simulation can emit `help_call` from an exact transcript
match. A future UNO Q integration must use one microphone stream, fan its
non-overlapping frames to this ASR engine, and separately retain the rolling one-second
YAMNet window.

- Source: the `sherpa-onnx-streaming-zipformer-en-2023-06-26` asset from
  [Sherpa-ONNX's `asr-models` release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models),
  converted from
  [`Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17`](https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17).
- License: the upstream model card has no license metadata. The Icefall and Sherpa
  inference code are Apache-2.0, but the checkpoint's redistribution terms must be
  confirmed before it is bundled into an App Lab package or otherwise redistributed.
  The upstream model has 66.11M parameters and reports 2.83% WER on LibriSpeech
  test-clean with beam search; this is not a claim about this project's microphone or
  deployment room.
- Local directory (ignored): `sherpa-onnx-streaming-zipformer-en-2023-06-26`.
  The verified SHA-256 hashes are `5022b2eca5b19d1bc104fcf33e26bc32604b7df553cd2e1f62e31dc7b05e9c87`
  for the int8 encoder, `85914f840b71110487ece7a30ebd0bd4b8fb5c2623215bba3aa5ab4c138fc591`
  for the FP32 decoder, and `abd5e30f3f16fc510605c6029dba33f10e4386bd75c5bdc30cf94076864db10d`
  for the int8 joiner.
- Input: non-empty, mono `float32` waveform frames at 16 kHz, normalized to roughly
  `[-1, 1]`. Sherpa computes 80-bin acoustic features internally and accepts each
  captured frame once.
- Assets: `encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx` (70,108,816 bytes),
  `decoder-epoch-99-avg-1-chunk-16-left-128.onnx` (2,093,080 bytes),
  `joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx` (259,416 bytes), and
  `tokens.txt` (the BPE token order). The adapter uses CPU-only greedy decoding and
  one compute thread.
- Output: changed partial text and endpointed final text. It has no calibrated class
  probabilities, label order, or decision threshold.
- Laptop smoke test: the packaged `test_wavs/0.wav` produced the final text
  `AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS`
  with `sherpa-onnx==1.13.8` on 2026-09-23. This only proves loading and inference;
  microphone accuracy, latency, memory, and UNO Q compatibility remain unmeasured.
- Laptop `help` simulation: `TranscriptHelpYAMNetInferenceEngine` detects exact,
  standalone `help` in changed Zipformer partial or final text and emits the existing
  `help_call` event with confidence `1.0`. This is a deterministic text match, not a
  calibrated ASR confidence or an emergency-use safety claim. It is connected only to
  the laptop `MockHardware` configuration. With the local
  `runs/recordings/help/recording-20260922-172120.wav` fixture followed by one second
  of silence, it emitted `help_call` with transcript `HELP`; that is a smoke test, not
  an accuracy evaluation.

## Local help keyword prototype

`help-kws.onnx` is generated locally and remains git-ignored. It embeds the Apache-2.0
YAMNet graph documented above, aggregates YAMNet's 1024-element embeddings with mean
and maximum pooling, and appends a standardized logistic classifier plus sigmoid.
The learned classifier is trained by this repository; its synthetic Piper training
speech and LibriSpeech examples are recreated using [the data instructions](../data/README.md).
No separately licensed third-party keyword checkpoint is included.

- Local filename: `help-kws.onnx`; current local SHA-256:
  `6f878423237416b5005bdfb4fe52ee8c1385ae458d8ded33cc701d078c4c058f`.
- Input: rank-1 `float32` tensor `waveform`, one second of mono 16 kHz PCM in
  `[-1, 1]`. YAMNet's log-mel frontend is part of the graph.
- Output: rank-1 tensor `help_probability` with one sigmoid probability.
- Labels: `0` is background and the single output represents `help_call`.
- Configured activation threshold: `0.62`, selected on a held-out mix of synthetic
  speech and full-length LibriSpeech background clips using live-style overlapping
  windows.

Build the local model after following `data/README.md`:

```bash
uv pip install --python .venv/bin/python onnx==1.19.0
uv run python scripts/train_help_kws.py \
  --dataset data/help-kws/synthetic \
  --additional-train-background data/help-kws/librispeech-background/train \
  --additional-train-help data/help-kws/librispeech-help-train \
  --additional-validation-background data/help-kws/librispeech-background/validation \
  --yamnet models/yamnet.onnx --output models/help-kws.onnx
```

The current held-out LibriSpeech `test-clean` result at threshold `0.62` is 6 true
positives, 21 false positives, 39 true negatives, and 9 false negatives across 75
utterances (precision 0.222, recall 0.400, accuracy 0.600). This is an honest
prototype result, not a safety-ready detector. The test utterances contain the word
in audiobook prose rather than urgent calls, but the false-positive and false-negative
rates are still too high. Do not rely on this model for emergencies. The next model
iteration needs genuine, consented near/far microphone recordings of urgent “help”
calls and confusing household speech; record a new untouched test split before
retuning or claiming improvement.
