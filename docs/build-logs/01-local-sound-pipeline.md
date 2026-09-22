# Build log 1: local sound pipeline and privacy boundary

Date: 2026-09-22

The first challenge-specific pass replaced the scalar-only demo path with a local
audio pipeline while preserving the repository's Input -> Preprocessing -> Inference
-> Decision -> Hardware boundary.

Implemented and verified on the development laptop:

- PCM WAV replay, deterministic synthetic sound generation, and a live PortAudio
  microphone adapter suitable for selecting the MOVO USB-M1;
- resampling, fixed-window preparation, and compact spectral features;
- distinct generated signatures for smoke alarm, breaking glass, and fall-like thud;
- temporal confirmation, alert hold time, and notification cooldowns;
- three 8x8 icon patterns and high-level UNO Q Bridge calls;
- a metadata-only SMTP notification boundary with asynchronous failure isolation.

The transparent spectral classifier exists to test every integration layer before a
trained model is selected. It correctly separates the deterministic fixtures, but it
is not evidence of real-world accuracy. The next model step is to collect or source
licensed evaluation sounds, benchmark candidate local models, and document the chosen
model's output activation and label mapping.

Hardware work still requiring genuine board evidence:

- compile the App Lab sketch with the installed Bridge and `Arduino_LED_Matrix`
  versions;
- visually confirm icon orientation and centering on the onboard 8x13 matrix;
- verify the USB microphone name, sample-rate support, and sustained capture;
- measure end-to-end latency and classification behavior in the demo room.

No build photos or physical measurements are claimed in this log because that work
has not yet been performed.
