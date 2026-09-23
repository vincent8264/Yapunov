# Private Local Sound Alerts (Arduino UNO Q)

This project detects safety-relevant sounds without a camera or cloud audio. Audio is
captured and classified locally; only an event name, confidence, device name, and
timestamp can leave the device when optional email notifications are enabled.

The event set is smoke alarm, breaking glass, a fall-like thud, and an optional
spoken `help` keyword detector. Four distinct 8x8 icons are centered on the UNO Q's
onboard 8x13 LED matrix.

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

During development, `MockHardware` records the selected visual alert. On the UNO Q,
`UnoQHardware` sends high-level event names across the Bridge and the STM32 renders
the corresponding matrix icon. In normal operation the configured live sound display
sends a derived 13-band audio spectrum at 20 Hz; raw microphone frames never
enter the notification or hardware APIs. Confirmed alert icons take priority over the
idle spectrum.

## Setup

Install [uv](https://docs.astral.sh/uv/), then on Windows or macOS run:

```bash
uv sync
```

Run the hardware-free demo:

```bash
uv run edge-ai run --config configs/demo.toml
```

Run the deterministic three-sound integration demo:

```bash
uv run edge-ai run --config configs/sound-demo.toml --max-steps 6
```

This generates actual waveforms, extracts spectral features, classifies the three
signatures, and records the visual decisions. Its transparent spectral rules validate
the complete pipeline; they are not a production safety model.

Pick a different synthetic sound on every step and print the 8x8 icon the matrix
would show:

```bash
uv run edge-ai run --config configs/sound-random.toml --max-steps 6
```

On the UNO Q, import the packaged app described under "Adding the UNO Q". It runs
these same random sounds and sends each alert to the onboard matrix.

For live YAMNet inference, first download the model once (the application does not
download anything at runtime):

```bash
curl -L https://huggingface.co/audiomagic/yamnet-onnx/resolve/main/yamnet.onnx \
  -o models/yamnet.onnx
```

Then select the MOVO USB-M1 as the operating system's default input and run:

```bash
uv run edge-ai run --config configs/sound-live.toml
```

To see the 13×8 matrix on the PC instead of printing `MOCK` spectrum rows, use the
desktop preview (close its window or press Ctrl+C to stop):

```bash
uv run edge-ai run --config configs/sound-live-preview.toml
```

To see what the unmodified YAMNet model thinks a recording contains, independently
of the project's three-event mapping, run:

```bash
uv run edge-ai inspect-audio sample.wav \
  --model models/yamnet.onnx \
  --class-map models/yamnet_class_map.csv \
  --top-k 10
```

This prints YAMNet's raw AudioSet labels and scores. It is useful for examining
speech, alarms, impacts, and confusing household sounds, but YAMNet does not detect
the meaning of the word `help`.

If the microphone is not the default, add its verified name or index as `device` in
the `[input]` section. On Linux, the `sounddevice` package also requires the system's
PortAudio runtime.

### Capture a microphone recording

Use the included recorder to make a WAV fixture for local testing. It records 16 kHz,
mono, 16-bit PCM by default, matching YAMNet's expected input rate:

```bash
uv run python scripts/record_microphone.py
```

Press Enter when you are ready to record, then press Ctrl+C to stop and save the WAV
file. The default destination is `runs/recordings/recording-<timestamp>.wav`; choose a
different source or destination with, for example:

```bash
uv run python scripts/record_microphone.py --device "MOVO USB-M1" --output runs/doorbell.wav
```

The live configuration uses `[display]` to sample 50 ms audio chunks at 20 Hz and
render their 13-band FFT spectrum on the 13×8 matrix. It accumulates those same
chunks into the unchanged one-second YAMNet inference window. Adjust `floor_db` and
`ceiling_db` only after measuring the deployment room; the display contains no raw
audio and does not affect classification. Confirmed danger icons stay visible for
five seconds in the live configurations before the spectrum resumes.

Stop it with Ctrl+C, or run a fixed number of iterations:

```bash
uv run edge-ai run --config configs/demo.toml --max-steps 5
```

The TOML file selects the input, preprocessing, inference, decision policy, hardware
backend, and loop interval. WAV and model paths are resolved relative to the config.

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
`--allow-actuators`. The UNO Q Bridge and sketch endpoints still require validation
on the physical board.

## Adding a real model

1. Put the `.onnx` file in `models/`.
2. Copy `configs/demo.toml`, set `[inference].type = "onnx"`, and add a `model`
   path relative to that config file (for example, `../models/model.onnx`). Set
   `output_type = "probabilities"` only after confirming that the model includes its
   output activation. Otherwise, add a model-specific output adapter in code.
3. Select `audio_waveform` preprocessing and match its sample rate and duration to the
   model, or add a model-specific spectrogram adapter.
4. If the model output is not a scalar or class-score vector, pass a model-specific `output_adapter` to `ONNXInferenceEngine`. This keeps conversion at the model boundary.

No model is required for the demo or tests.

The included `spectral_demo` engine remains a deterministic test baseline. The live
configuration instead uses a YAMNet-specific ONNX adapter that consumes a raw 16 kHz
mono waveform and reduces the model's 521 AudioSet outputs into the three project
events. Before a field demo, evaluate it on genuine recordings from the MOVO
microphone and deployment room and record its confusion matrix and thresholds.
ONNX Runtime telemetry is explicitly disabled before model sessions are created.

## Optional `help` keyword spotting

`configs/help-live.example.toml` adds a dedicated local keyword model without running
two microphone streams. The microphone contributes a 200 ms chunk to one rolling
one-second waveform on each step. The keyword detector runs on every step; YAMNet
runs every third step. The models run sequentially so their peak CPU work does not
overlap. Each detector's latency is included in runner output as `keyword_ms` and,
when scheduled, `environment_ms`.

The repository supplies and tests the scheduler and ONNX adapter, but does **not**
include or claim a trained `help` model. Before using the example config, supply
`models/help-kws.onnx` with the exact contract documented in `models/README.md`, then
measure latency on the UNO Q and tune `environment_every_steps` if inference falls
behind the 200 ms capture cadence.

Build a held-out test set with short PCM WAV clips in this layout:

```text
keyword-test/
  help_call/
    speaker-a-01.wav
  background/
    conversation-01.wav
```

Include multiple speakers, distances, room noise, television speech, and hard
negatives such as “health” and “hello.” Keep training speakers out of this folder.
Evaluate the same exported model and threshold used by the device with:

```bash
uv run edge-ai evaluate-keyword \
  --dataset keyword-test \
  --model models/help-kws.onnx \
  --threshold 0.5
```

The command reports TP/FP/TN/FN, precision, recall, and accuracy. The clips are
resampled to 16 kHz and padded or trimmed to one second. Accuracy alone is not enough
for this use case; separately inspect false negatives and television/conversation
false positives before enabling family notifications.

## Sound event policy

The `sound_events` decision policy supports:

- a separate confidence threshold for every event;
- per-event confirmation counts, allowing sustained confirmation for alarms and a
  single window for transient glass/impact events;
- a visual hold time so an icon remains readable;
- per-event notification cooldowns to avoid repeated messages.

Background or unknown model labels do not trigger an alert. “Unfamiliar voice” is not
claimed: identity-aware voice recognition needs consent, enrollment, and separate
validation. A later extension can safely detect generic speech during configured quiet
hours without identifying a speaker.

## Optional email

Create the ignored private configuration from the checked-in template:

```bash
cp configs/private-live.example.toml configs/private-live.toml
```

Then edit the SMTP server, sender, recipient, device name, and timezone in
`configs/private-live.toml`. Its notification section looks like this:

```toml
[notifications]
type = "smtp"
settings_file = "../data/notification-settings.json"
device_name = "Living room sound monitor"
host = "smtp.example.com"
port = 587
sender = "monitor@example.com"
recipient = "family@example.com"
username = "monitor@example.com"
password_env = "EDGE_AI_SMTP_PASSWORD"
starttls = true
timeout_seconds = 5.0
timezone = "America/Los_Angeles"
status_log = true
```

The private file is explicitly ignored by Git. Set the named password environment
variable separately rather than writing the password into TOML:

```bash
export EDGE_AI_SMTP_PASSWORD='your-smtp-app-password'
```

In PowerShell, use
`$env:EDGE_AI_SMTP_PASSWORD = 'your-smtp-app-password'` instead. Email work runs on
a background queue; network failure cannot prevent the local matrix alert. No
waveform, audio feature, or recording path is present in the notification data
structure.
Messages use fixed, event-specific wording; no LLM or other text-generation service is
involved. Smoke alarms request an immediate check, while glass and fall-like impacts
are explicitly described as possible events rather than confirmed emergencies.

Before the demo, send one test email without starting the microphone, model, or
hardware:

```bash
uv run edge-ai test-notification --config configs/private-live.toml --event smoke_alarm
```

The command exits with an error if delivery fails. During normal operation, successful
and failed background deliveries are printed without exposing credentials or audio.

### Local notification setup portal

Once the private SMTP configuration and password environment variable are ready,
start the temporary setup portal on the computer:

```bash
uv run edge-ai setup --config configs/private-live.toml
```

Open the printed local address. Computer-only setup on `127.0.0.1` does not require a
PIN. When the portal is exposed over the LAN, enter the six-digit one-time PIN shown
in the terminal. The page collects up to five comma-separated recipient emails, the
device/room name, timezone, and whether remote notifications are enabled. **Save and
send test** verifies SMTP delivery to every recipient before saving the preferences.

When the command runs on the UNO Q, use `--host 0.0.0.0` and open the board's LAN IP
from the setup computer:

```bash
uv run edge-ai setup --config configs/private-live.toml --host 0.0.0.0
```

The LAN PIN prevents another person on the same network from changing recipients or
triggering a test email during setup. The portal stores only non-secret preferences in
the configured JSON file with
owner-only permissions. The SMTP password stays in the environment, and the portal
stops when you press Ctrl+C. The regular `edge-ai run` command automatically uses the
saved preferences. Persistence across an App Lab redeploy and unattended startup must
still be verified on the physical UNO Q before field use.

## Adding the UNO Q

App Lab deploys one app, not this repository. Build that app, then import the zip:

```bash
uv run python scripts/package_app_lab.py
```

This writes `dist/private-sound-alerts.zip`. The archive root contains `app.yaml`,
the sketch, and a Python bundle of the random-sound demo.

Before adding a microphone, validate the real bundled YAMNet model with deterministic
WAV fixtures:

```bash
uv run python scripts/package_app_lab.py --mode yamnet-test
```

Import `dist/private-sound-alerts-yamnet-test.zip` in App Lab. This larger package
installs ONNX Runtime and runs four bundled WAV files through the real YAMNet model.
It should recognize the synthetic alarm and glass fixtures. The fall-like fixture is
classified as `fall_thud`, but at about 0.10, below the 0.20 alert threshold, so it
does not raise an alert. This is a model-loading
and inference smoke test, not accuracy evidence, and it does not open a microphone.

In Arduino App Lab, with the UNO Q connected by USB-C and its first-time setup
already finished:

1. Open **My Apps**.
2. Choose **Create new app**, then **Import App**.
3. Import `dist/private-sound-alerts.zip`.
4. Open the imported app and run it.

App Lab compiles `sketch/sketch.ino` onto the board's microcontroller and starts
`python/main.py` on the Linux side. Each loop synthesizes one random sound, classifies
it locally, and calls `show_alert` so the onboard matrix draws that icon. The App Lab
log shows lines such as `label=glass_break`. This build does not open the microphone,
load YAMNet, or send email.

For real sound classification on the board, download `models/yamnet.onnx` as
described above and package the live configuration:

```bash
uv run python scripts/package_app_lab.py --config configs/sound-uno-q-live.toml
```

This writes `dist/private-sound-alerts-live.zip` (about 16 MB), which bundles YAMNet
and its class map and adds `onnxruntime` to the board's requirements. The app reads
the first USB microphone through App Lab's ALSA microphone peripheral. The UNO Q's
single USB-C port must then carry a powered USB-C hub with the microphone attached,
so run App Lab in Network Mode with the board on Wi-Fi. The first start also needs
internet to install `onnxruntime`. The log prints YAMNet's top AudioSet label on each
line (`model_label=Alarm`), which helps tune the thresholds in the live config.

To exercise the same YAMNet, decision, matrix, and email chain without a microphone,
package `configs/sound-uno-q-yamnet-sim.toml` instead. It feeds random synthetic
smoke-alarm, glass, thud, and background clips into the real model and writes
`dist/private-sound-alerts-yamnet-sim.zip`. The synthetic clips are not recordings:
on the laptop YAMNet recognized the alarm in 30 of 30 clips, glass in 27 of 30, and
the thud in 26 of 30, but individual glass and thud scores sometimes fall below the
0.20 threshold, so not every generated clip raises an alert.

To send email from the board as well, first confirm delivery from the laptop with
`edge-ai test-notification`, then build the email variant:

```bash
export EDGE_AI_SMTP_PASSWORD='your-smtp-app-password'
uv run python scripts/package_app_lab.py --notifications configs/private-live.toml
```

This writes `dist/private-sound-alerts-email.zip`. It copies the `[notifications]`
settings into the board config and stores the password in `python/smtp-password`,
because App Lab does not pass laptop environment variables to the board. That zip
contains the secret: import it only on your own UNO Q and never upload it to the
project page. The board must be on Wi-Fi with outbound SMTP allowed. Email runs in the
background, so a failed send is logged and never blocks the matrix alert.

The sketch uses the documented onboard matrix library and no external pins. Confirm
the installed App Lab, Bridge, and `Arduino_LED_Matrix` versions on the board, and
check that each icon is upright and centered. External PWM and servo paths remain
deliberately disabled.
