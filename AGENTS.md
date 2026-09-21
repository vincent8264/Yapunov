# AI Agent Guide

This repository prepares for a challenge that remains unknown until the hackathon
begins. Keep changes reusable, small, hardware-free to test, and easy to replace once
the actual problem and supplied components are known.

## Architecture

Preserve the core flow:

```text
Input -> Preprocessing -> Inference -> Decision -> Hardware
```

Add challenge-specific behavior as an adapter or decision function instead of
embedding it in the runner. Laptop development must continue to work with simulated
input and `MockHardware`.

## Standard workflow

```bash
uv sync
uv run pytest
uv run edge-ai run --config configs/demo.toml --max-steps 5
uv run edge-ai hardware-check --config configs/hardware-check.toml
```

Before handing off changes, run relevant tests and one short configured demo. Runtime
must not require network access after dependencies and models are installed.

## Configuration

- Use TOML files under `configs/`; avoid adding a YAML dependency.
- Keep component factories explicit in `src/edge_ai/config.py`. Do not build a dynamic
  plugin system unless the challenge clearly needs one.
- Fail at startup with useful messages for missing sections, unsupported types, bad
  ranges, missing models, and incompatible settings.
- Resolve referenced paths relative to the configuration file.
- Never commit secrets, machine-specific absolute paths, or guessed hardware pins.
- When adding a component type, update the factory, add a minimal example, and test
  successful and invalid construction.

## Hardware safety

- Do not guess UNO Q pins, voltage levels, Bridge versions, or actuator libraries.
- Keep hardware imports local so laptop imports and tests continue to work.
- Reject unsafe ranges before hardware calls and leave outputs safe/off on shutdown,
  interruption, and failure when possible.
- Test Bridge behavior through dependency injection or fakes. Clearly label anything
  that still requires testing on the actual board.
- Keep actuator diagnostics opt-in. Add checks only after verifying wiring, voltage,
  pin assignments, and mechanical travel; always restore switchable outputs to safe
  values.
- Use only event-supplied hardware, apart from the permitted laptop and USB-C cable.

## Models and data

- For each selected model, document its source, license, input shape/layout/dtype,
  normalization, label order, and output interpretation in `models/README.md`.
- Never assume ONNX outputs are probabilities; confirm activation requirements.
- Keep samples free of credentials and unnecessary personal data. Use deterministic
  seeds and small fixtures where practical.

## Tests and documentation

Cover happy paths, invalid configuration, boundaries, unavailable inputs, cleanup,
and mocked hardware calls. Do not weaken a test without documenting the intended
behavior change.

The Hackaday.io page is part of the judged deliverable. Record design reasoning,
failed attempts, wiring changes, measurements, and limitations as the project evolves.
The final page needs at least three useful build logs, five genuine build/final photos,
and three uploadable project files. Never fabricate build evidence.
