# Edge AI Hackathon Starter (Arduino UNO Q)

This repository provides a small, reusable Edge AI architecture for an Arduino UNO Q hackathon. It runs entirely on a Windows or macOS laptop today, without a board, camera, model, or network connection after dependencies are installed.

## Architecture

```text
Input
  ↓
Preprocessing
  ↓
Inference
  ↓
Decision
  ↓
Hardware
```

During development, `MockHardware` records and prints actions. On the UNO Q, the same pipeline can instead use `UnoQHardware`, which will send high-level decisions across the UNO Q Bridge to the STM32. Similarly, `DummyInferenceEngine` runs immediately without a model, while `ONNXInferenceEngine` loads a real ONNX model.

## Setup

Install [uv](https://docs.astral.sh/uv/), then on Windows or macOS run:

```bash
uv sync
```

Run the hardware-free demo:

```bash
uv run edge-ai run --config configs/demo.toml
```

Stop it with Ctrl+C, or run a fixed number of iterations:

```bash
uv run edge-ai run --config configs/demo.toml --max-steps 5
```

The TOML file selects the input, preprocessing, inference, decision, hardware backend,
and loop interval. Copy `configs/demo.toml` when creating a challenge-specific setup;
relative model paths are resolved from the config file's directory.

Run the tests with:

```bash
uv run pytest
```

## Hardware check

Run the safe mock check on a laptop:

```bash
uv run edge-ai hardware-check --config configs/hardware-check.toml
```

During the event, copy the config, select `uno_q`, and add only verified component
checks. PWM and servo checks are skipped unless explicitly enabled with
`--allow-actuators`. Save a machine-readable report with `--report runs/hardware.json`.
The UNO Q Bridge and sketch endpoints still require validation on the physical board.

## Adding a real model

1. Put the `.onnx` file in `models/`.
2. Copy `configs/demo.toml`, set `[inference].type = "onnx"`, and add a `model`
   path relative to that config file (for example, `../models/model.onnx`).
3. Adapt image/sensor preprocessing to the model's expected shape and dtype.
4. If the model output is not a scalar or class-score vector, pass a model-specific `output_adapter` to `ONNXInferenceEngine`. This keeps conversion at the model boundary.

No model is required for the demo or tests.

## Adding the UNO Q

When the board arrives:

1. Open and test `app_lab/starter_app` in Arduino App Lab.
2. Verify the installed App Lab and Bridge API versions against the board image.
3. Implement and test the Bridge calls in `UnoQHardware` (the current methods are deliberately guarded and untested).
4. Verify GPIO, PWM, servo pins, voltage levels, and communication with the STM32 using safe test hardware.
5. Set `[hardware].type = "uno_q"` in the selected config; the rest of the pipeline stays unchanged.

The App Lab files follow Arduino's current documented structure and RPC concepts, but cannot be built or hardware-tested without an UNO Q. Pin choices and actuator wiring are intentionally left for the hackathon.
