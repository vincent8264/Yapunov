# Local audio data

Audio datasets, generated clips, and evaluation recordings under this directory are
local-only and git-ignored. They can be recreated with the commands below. Do not
treat the proxy clips as evidence that the system detects real emergencies.

## Help keyword data

The synthetic set uses Piper's `en_US-libritts_r-medium` voice. The voice model is
trained from LibriTTS-R (CC BY 4.0); Piper is used only as a local generation tool.

```bash
mkdir -p /tmp/edge-ai-piper
curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx \
  -o /tmp/edge-ai-piper/en_US-libritts_r-medium.onnx
curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json \
  -o /tmp/edge-ai-piper/en_US-libritts_r-medium.onnx.json
uv venv /tmp/edge-ai-kws-tools --python 3.13
uv pip install --python /tmp/edge-ai-kws-tools/bin/python piper-tts==1.8.0 soundfile==0.13.1 onnx==1.19.0
/tmp/edge-ai-kws-tools/bin/python scripts/generate_help_speech.py \
  --voice /tmp/edge-ai-piper/en_US-libritts_r-medium.onnx \
  --output data/help-kws/synthetic
```

The real-speech training/background and held-out test sets come from LibriSpeech
`dev-clean` and `test-clean` (CC BY 4.0). Dev-clean contributes real-human negative
clips and 15 approximately aligned `help` examples; test-clean remains the unseen
evaluation split.

```bash
curl -L https://www.openslr.org/resources/12/dev-clean.tar.gz -o /tmp/dev-clean.tar.gz
curl -L https://www.openslr.org/resources/12/test-clean.tar.gz -o /tmp/test-clean.tar.gz
tar -xzf /tmp/dev-clean.tar.gz -C /tmp
tar -xzf /tmp/test-clean.tar.gz -C /tmp

/tmp/edge-ai-kws-tools/bin/python scripts/prepare_librispeech_help_eval.py \
  --librispeech /tmp/LibriSpeech/dev-clean \
  --output data/help-kws/dev-reference \
  --background-output data/help-kws/librispeech-background \
  --training-help-output data/help-kws/librispeech-help-train

/tmp/edge-ai-kws-tools/bin/python scripts/prepare_librispeech_help_eval.py \
  --librispeech /tmp/LibriSpeech/test-clean \
  --output data/help-kws/test-clean \
  --background-output data/help-kws/test-clean-unused \
  --test-background-count 60 --train-background-count 0 \
  --validation-background-count 0
```

## Environmental sound samples

Create `data/evaluation/glass_break/` and `data/evaluation/fall_thud/`, then download
these PCM WAV files from the ESC-50 repository:

- Glass breaking (CC BY-NC 3.0):
  `1-20133-A-39.wav`, `1-84536-A-39.wav`, `1-84704-A-39.wav`
- Impact proxies (ESC-50 `door_wood_knock`, CC BY-NC 3.0):
  `1-101336-A-30.wav`, `1-103995-A-30.wav`, `1-103999-A-30.wav`

Each file is available at:

```text
https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/<filename>
```

The smoke-alarm sample is a CC0 recording credited in the source repository's
`SOURCES.md` to SpliceSound. Download it to `data/evaluation/smoke_alarm/`:

```text
https://raw.githubusercontent.com/vighriday/stay-for-dogs/main/public/test-audio/control/369847-smoke-detector-alarm-distant-neighbori.mp3
```

Convert the MP3 to uncompressed PCM WAV before feeding it to this project (the local
setup above includes SoundFile):

```bash
/tmp/edge-ai-kws-tools/bin/python -c 'import soundfile as s; x,r=s.read("data/evaluation/smoke_alarm/369847-smoke-detector-alarm-distant-neighbori.mp3"); s.write("data/evaluation/smoke_alarm/369847-smoke-detector-alarm-distant-neighbori.wav",x,r,subtype="PCM_16")'
```

The door knocks are only impact/thud proxies, not recorded human falls, and one
smoke-detector clip is not a representative test set.
