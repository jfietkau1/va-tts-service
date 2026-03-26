# VA TTS Service

A Python service that receives text commands over WebSocket, synthesizes speech using XTTS v2, and plays audio through the speaker.

## Architecture

```
WebSocket (:8766) → Speech Queue → XTTS v2 (inference_stream) → sounddevice → Speaker
```

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (required for XTTS)
- A reference audio file for each voice (3-10 seconds of speech)

## Setup

```bash
git clone https://github.com/jfietkau1/va-tts-service.git
cd va-tts-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add at least one voice reference file (see "Adding Voices" below)
```

## Adding Voices

Drop a 3-10 second `.wav` file into a subdirectory of `voices/`:

```
voices/
  default/
    reference.wav    # Your default voice
  morgan/
    reference.wav    # Another voice
  french/
    reference.wav    # A French speaker
```

On startup, the service pre-computes speaker embeddings for each voice.
The orchestrator (via OpenClaw) can list and select voices at runtime.

## Running

```bash
python -m tts_service.main
```

## WebSocket Protocol

### Commands (received from orchestrator)

| Command | Description |
|---|---|
| `{ "type": "speak", "text": "...", "voice": "default", "language": "en" }` | Synthesize and play speech |
| `{ "type": "stop" }` | Interrupt current playback |
| `{ "type": "config", "voice": "...", "language": "..." }` | Update default voice/language |
| `{ "type": "listVoices" }` | List available voices |

### Status (sent to orchestrator)

| Status | Description |
|---|---|
| `{ "type": "speaking", "text": "..." }` | Started speaking a chunk |
| `{ "type": "idle" }` | Finished all queued speech |
| `{ "type": "stopped" }` | Playback was interrupted |
| `{ "type": "voices", "voices": [...] }` | Available voice list |

## Configuration

| Variable | Description | Default |
|---|---|---|
| `XTTS_MODEL_PATH` | Path to XTTS model (empty = auto-download) | (auto) |
| `XTTS_DEVICE` | Device: cuda/cpu | `cuda` |
| `XTTS_USE_DEEPSPEED` | Enable DeepSpeed optimization | `true` |
| `VOICES_DIR` | Directory containing voice reference files | `./voices` |
| `DEFAULT_VOICE` | Default voice ID | `default` |
| `DEFAULT_LANGUAGE` | Default language code | `en` |
| `WS_HOST` | WebSocket bind address | `0.0.0.0` |
| `WS_PORT` | WebSocket port | `8766` |
| `AUDIO_DEVICE_INDEX` | Speaker device index (-1 = default) | `-1` |
