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

For live YAMNet inference, first download the model once (the application does not
download anything at runtime):

```bash
curl -L https://huggingface.co/audiomagic/yamnet-onnx/resolve/main/yamnet.onnx \
  -o models/yamnet.onnx
```

Model binaries and downloaded/generated audio are deliberately excluded from Git.
[models/README.md](models/README.md) documents the model contracts and local help
model build; [data/README.md](data/README.md) gives exact Piper, LibriSpeech, ESC-50,
and smoke-alarm download locations and licenses. The current `help-kws.onnx` is a
real locally trained YAMNet-transfer prototype, but its held-out real-speech result is
not good enough for emergency use; review the recorded false-positive and
false-negative counts before enabling it.

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
chunks into a one-second YAMNet window, then retains the most recent window and runs
again every 200 ms (five inferences per second). Set `inference_hop_seconds` equal
to the window duration, or omit it, to retain the original non-overlapping behavior.
Adjust `floor_db` and `ceiling_db` only after measuring the deployment room; the
display contains no raw audio and does not affect classification. Confirmed danger
icons stay visible for five seconds in the live configurations before the spectrum
resumes.

The canonical UNO Q configuration also enables conservative microphone health
checks. Five seconds of exact digital zeroes is reported as `no_signal`, while an
exactly repeated nonzero capture buffer is reported as `frozen_signal`. Read errors
are reported as `unavailable`. These observations can indicate a muted, disconnected,
or stalled input, but they do not prove that the microphone hardware is broken. The
zero-level threshold deliberately avoids guessing the noise floor before measurements
are taken on the event microphone; increase it only after room calibration.

After a runtime fault, the pipeline remains alive and displays a crossed-microphone
icon. On the UNO Q it continues polling the existing App Lab microphone object so
App Lab's ALSA adapter can perform its own USB hot-plug retries. The desktop input is
recreated every two seconds instead. Recovery requires a usable nonzero frame; a
frozen-input recovery also requires the next frame to change. With SMTP notifications
enabled, the first fault sends one metadata-only warning, the matrix status column
lights after delivery, and a later recovery sends one follow-up email. Repeated retry
failures do not generate duplicate messages.

If the microphone is absent or still claimed by an earlier process when the app
starts, the health wrapper also keeps the pipeline alive and retries opening it every
two seconds. This startup case uses the same fault icon and notification transition
instead of terminating the App Lab container.

For the physical demo, start with the live spectrum visible, unplug only the USB
microphone, and leave the UNO Q and powered hub connected. Verify the fault icon and
warning email, reconnect the microphone, make a short sound, and verify that the
spectrum and inference resume before checking the recovery email. USB removal and
re-enumeration still require validation against the App Lab version and event hardware;
do not claim reconnection until this exact sequence has passed on the board.

Stop it with Ctrl+C, or run a fixed number of iterations:

```bash
uv run edge-ai run --config configs/demo.toml --max-steps 5
```

The TOML file selects the input, preprocessing, inference, decision policy, hardware
backend, and loop interval. WAV and model paths are resolved relative to the config.

### Configuration files

Each committed TOML has one distinct role. Reuse these files instead of creating
copies for temporary experiments:

| File | Purpose |
| --- | --- |
| `demo.toml` | Generic, hardware-free sensor pipeline used to verify the reusable architecture. |
| `hardware-check.toml` | Safe laptop hardware-check plan using `MockHardware`; actuator checks remain opt-in. |
| `sound-uno-q.toml` | Canonical UNO Q deployment: board microphone input, YAMNet, decision policy, and real matrix output. This is the only committed UNO Q pipeline config. |
| `private-live.example.toml` | Template for a private SMTP-enabled configuration. Copy it to ignored `private-live.toml` and never commit credentials. |
| `help-live.example.toml` | Optional scheduled YAMNet plus `help` keyword configuration; it requires a separately validated keyword model. |

Test-specific variations should be constructed in tests, generated by packaging
scripts, or kept in a temporary untracked file. Add another committed TOML only when
it represents a genuinely different user-facing workflow, and add it to this table.

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

### Developing-risk warnings

Optional `[decision.risks.<name>]` tables add warnings for risks that build up over
time. They are tracked separately from the emergency events and never change an
emergency alert. `configs/sound-uno-q.toml` defines `water_leak`, which listens for
`Drip`, `Trickle, dribble`, `Gush`, and `Water tap, faucet`.

YAMNet reports its raw score for each listed AudioSet label. A timestamped history
keeps only the last `window_seconds` (default 15 s). Each inference is credited with the time
since the previous one, capped at the one-second model window. A warning such as
`[WARNING] Possible water leak detected` prints only when any listed label scores at
or above `threshold` for at least `min_detected_seconds` (default 7 s) of that window, and then
not again for `cooldown_seconds` (default 300 s). Startup fails if a label is not in the
YAMNet class map or if the inference type is not YAMNet. When email is configured, each
printed warning also sends one background email with the detected duration and
strongest label; `cooldown_seconds` therefore also limits repeat emails. Warnings do
not change the matrix or its email-status pixel. The 0.15 thresholds are uncalibrated;
tune them with real recordings of the tap, a leak, and cooking in the deployment room.

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

The private file is explicitly ignored by Git. The preferred approach is to set the
named password environment variable separately. For a single-file local setup, the
packager also accepts `password = "your-app-password"` in this ignored private file.
It removes that value from the packaged TOML and writes it to the private app's
`python/smtp-password` file instead. Never commit or share the private config or the
generated email ZIP.

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

For a manual UNO Q session, use `--host 0.0.0.0` and open the board's LAN IP from the
setup computer:

```bash
uv run edge-ai setup --config configs/private-live.toml --host 0.0.0.0
```

`0.0.0.0` is the server's listen address, not an address to enter in a browser. Open
`http://<board-ip>:8080`; for example, a board assigned `192.168.1.50` is reached at
`http://192.168.1.50:8080`. A router DHCP reservation is optional, but keeps that URL
from changing. A verified local DNS or mDNS hostname can be used instead. Do not
port-forward the setup portal or expose it to the public internet.

The LAN PIN prevents another person on the same network from changing recipients or
triggering a test email. The portal stores only non-secret preferences in the
configured JSON file with owner-only permissions. The SMTP password stays in the
environment, and a manual portal stops when you press Ctrl+C. The regular `edge-ai
run` command uses the saved preferences on its next start.

The email-enabled App Lab package starts the same portal in a background thread at
`0.0.0.0:8080` while sound detection continues. Its six-digit PIN is created in an
owner-only `setup-pin` file and printed in the App Lab log. Saving the form safely
replaces the running notifier, so new recipients, device name, timezone, and enabled
state take effect without restarting detection. The settings and PIN persist across
normal app restarts in the app directory; persistence across an App Lab redeploy must
still be verified on the physical UNO Q before field use.

## Adding the UNO Q

App Lab deploys one app, not this repository. Build that app, then import the zip:

```bash
uv run python scripts/package_app_lab.py
```

When `configs/private-live.toml` exists, this writes the email-enabled
`dist/private-sound-alerts-live-email.zip`; it contains that file's SMTP settings and
password. Keep the archive private. Without the private config, it writes
`dist/private-sound-alerts-live.zip`. The archive root contains `app.yaml`, the
sketch, the live UNO Q configuration, YAMNet, and its class map. To deliberately
build the non-email archive while the private config exists, add `--no-notifications`.

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
3. Import `dist/private-sound-alerts-live.zip`.
4. Open the imported app and run it.

App Lab compiles `sketch/sketch.ino` onto the board's microcontroller and starts
`python/main.py` on the Linux side. The app reads the first USB microphone through
App Lab's ALSA microphone peripheral, runs YAMNet locally, and calls `show_alert` so
the onboard matrix draws the detected event. The UNO Q's single USB-C port must carry
a powered USB-C hub with the microphone attached, so run App Lab in Network Mode with
the board on Wi-Fi. The first start also needs internet to install `onnxruntime`. The
log prints YAMNet's top AudioSet label on each line (`model_label=Alarm`), which helps
tune the thresholds in the live config.

Before monitoring begins, the live app confirms the UNO Q Bridge response and captures
one real microphone frame. It also makes a small HTTPS reachability check. A checkmark
is shown on the matrix for two seconds when the required checks pass; the rightmost bar
is lit when internet is reachable and unlit when it is not. Internet is advisory after
installation, so an offline board still begins local sound monitoring. A microphone or
Bridge failure stops startup and shows an X when the matrix remains reachable.

To send email from the board as well, first confirm delivery from the laptop with
`edge-ai test-notification`, then build the email variant:

```bash
export EDGE_AI_SMTP_PASSWORD='your-smtp-app-password'
uv run python scripts/package_app_lab.py --notifications configs/private-live.toml
```

This writes `dist/private-sound-alerts-live-email.zip`. The same private configuration
is selected automatically for normal packages when it exists; `--notifications` is
only needed to choose a different private config. The packager copies the
`[notifications]` settings into the board config and stores the password in
`python/smtp-password`, because App Lab does not pass laptop environment variables to
the board. That zip contains the secret: import it only on your own UNO Q and never
upload it to the project page. The board must be on Wi-Fi with outbound SMTP allowed.
Email runs in the background, so a failed send is logged and never blocks the matrix
alert. The App Lab log also prints the local setup URL and persistent setup PIN; open
the URL from a device on the same LAN to change recipients and send a test message.

The sketch uses the documented onboard matrix library and no external pins. Confirm
the installed App Lab, Bridge, and `Arduino_LED_Matrix` versions on the board, and
check that each icon is upright and centered. External PWM and servo paths remain
deliberately disabled.
