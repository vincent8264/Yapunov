# Private Local Sound Alerts

A camera-free sound monitor built for the Arduino UNO Q. It classifies microphone
audio on the device, shows a sound event on the onboard LED matrix, and can email
event metadata to a configured recipient. Audio recordings and features are not
part of the notification payload.

The project grew from a hackathon prompt about giving an older adult a visible
alert for a smoke alarm, breaking glass, or a fall-like impact. A `fall_thud`
classification indicates an impact sound; it cannot establish that a person fell.
This is a prototype, not an emergency detection or medical safety device.

## What the code does

```text
Microphone or simulated input -> preprocessing -> inference -> decision -> LED matrix or mock hardware
                                                   |
                                                   +-> optional metadata-only email
```

- A YAMNet ONNX adapter maps selected AudioSet classes to smoke alarm, glass
  break, and fall-like thud events. Per-event thresholds, confirmation counts,
  icon hold times, and notification cooldowns govern alerts.
- The UNO Q adapter sends event names across the Bridge. The sketch draws distinct
  icons on the 8×13 matrix; a 13-band audio spectrum fills idle time.
- A microphone health wrapper reports missing, silent, or frozen input and retries
  capture. Optional email alerts and a separate heartbeat watchdog report faults.
- The laptop path uses deterministic simulated input and `MockHardware`, so the
  pipeline and packaging logic can be exercised without a board.
- An optional `help` keyword scheduler shares the microphone stream with YAMNet.
  The locally trained keyword model is a research prototype with poor held-out
  results; it is not enabled in the canonical UNO Q configuration.

The software is covered by unit and integration tests using fakes for hardware,
SMTP, and model sessions. The committed record does not establish real-room
accuracy, end-to-end UNO Q latency, or reliable USB microphone reconnection on a
physical board. The live thresholds are starting values that need measurement.
See [model contracts and evaluation](models/README.md) for the recorded keyword
confusion matrix and other limitations.

## Run locally

Install [uv](https://docs.astral.sh/uv/) and Python 3.13 or newer, then run:

```bash
uv sync
uv run pytest
uv run edge-ai run --config configs/demo.toml --max-steps 5
uv run edge-ai hardware-check --config configs/hardware-check.toml
```

The demo and hardware check use no model download or physical board. After
dependencies and model files are installed, the runtime does not download data.
For live sound inference, download the pinned YAMNet model described in
[models/README.md](models/README.md) to `models/yamnet.onnx` and use the
[UNO Q deployment guide](docs/technical-guide.md#adding-the-uno-q). Model binaries,
recordings, private SMTP settings, and generated packages are excluded from Git.

## Configuration

Paths referenced inside a TOML file are relative to that file. The UNO Q config is
the single committed live board pipeline; example files supply optional private
settings or models.

| File | Purpose |
| --- | --- |
| `configs/demo.toml` | Hardware-free sensor pipeline for local development. |
| `configs/hardware-check.toml` | Safe hardware diagnostics using `MockHardware`; actuator checks are opt-in. |
| `configs/sound-uno-q.toml` | Canonical UNO Q microphone, YAMNet, decision, and matrix configuration. |
| `configs/private-live.example.toml` | Template for private SMTP settings and the optional heartbeat watchdog. |
| `configs/help-live.example.toml` | Optional YAMNet plus `help` keyword schedule; requires a separately built and validated model. |

## Project notes

- [Technical guide](docs/technical-guide.md): audio behavior, local email setup,
  watchdog, packaging, and App Lab instructions.
- [Models](models/README.md) and [data](data/README.md): source, license, input
  contracts, reproduction steps, and evaluation caveats.
- [Original challenge brief](PROBLEM.md) and
  [build logs](docs/build-logs/01-local-sound-pipeline.md): historical context.

The code intentionally keeps input, preprocessing, inference, decision, and
hardware behind separate interfaces. That made it possible to develop and test
the sound policy locally while keeping UNO Q imports inside the board adapter.
