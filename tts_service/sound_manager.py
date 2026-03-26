import logging
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PLAYBACK_SAMPLE_RATE = 24000


class SoundManager:
    """Loads and caches WAV sound effects from a sounds directory.

    Sound files are loaded into memory on startup for instant playback.
    Files are resampled to match the TTS output sample rate (24kHz).
    """

    def __init__(self, sounds_dir: str):
        self._sounds_dir = Path(sounds_dir)
        self._sounds: dict[str, np.ndarray] = {}

    def scan(self) -> list[str]:
        """Scan the sounds directory and pre-load all WAV files."""
        self._sounds.clear()

        if not self._sounds_dir.exists():
            logger.warning("Sounds directory does not exist: %s", self._sounds_dir)
            return []

        for wav_path in sorted(self._sounds_dir.glob("*.wav")):
            sound_id = wav_path.stem
            try:
                audio = self._load_wav(wav_path)
                self._sounds[sound_id] = audio
                logger.info("Loaded sound: %s (%.2fs)", sound_id, len(audio) / PLAYBACK_SAMPLE_RATE)
            except Exception:
                logger.exception("Failed to load sound: %s", wav_path)

        logger.info("Loaded %d sound(s)", len(self._sounds))
        return list(self._sounds.keys())

    def get(self, sound_id: str) -> np.ndarray | None:
        return self._sounds.get(sound_id)

    def _load_wav(self, path: Path) -> np.ndarray:
        with wave.open(str(path), "rb") as w:
            n_channels = w.getnchannels()
            sample_width = w.getsampwidth()
            sample_rate = w.getframerate()
            n_frames = w.getnframes()
            raw = w.readframes(n_frames)

        if sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        if sample_rate != PLAYBACK_SAMPLE_RATE:
            ratio = PLAYBACK_SAMPLE_RATE / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        return audio
