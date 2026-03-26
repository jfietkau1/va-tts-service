import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

XTTS_SAMPLE_RATE = 24000


@dataclass
class Voice:
    id: str
    name: str
    reference_path: Path
    languages: list[str] = field(default_factory=lambda: ["en"])
    gpt_cond_latent: torch.Tensor | None = None
    speaker_embedding: torch.Tensor | None = None


XTTS_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko",
]


class VoiceManager:
    """Scans a voices directory and pre-computes XTTS conditioning for each voice.

    Directory structure:
        voices/
          default/
            reference.wav    # 3-10 second reference clip
          morgan/
            reference.wav
    """

    def __init__(self, voices_dir: str):
        self._voices_dir = Path(voices_dir)
        self._voices: dict[str, Voice] = {}

    @property
    def voices(self) -> dict[str, Voice]:
        return self._voices

    def scan(self) -> list[Voice]:
        """Scan the voices directory for reference audio files."""
        self._voices.clear()

        if not self._voices_dir.exists():
            logger.warning("Voices directory does not exist: %s", self._voices_dir)
            return []

        for voice_dir in sorted(self._voices_dir.iterdir()):
            if not voice_dir.is_dir():
                continue

            ref_path = voice_dir / "reference.wav"
            if not ref_path.exists():
                ref_candidates = list(voice_dir.glob("*.wav"))
                if ref_candidates:
                    ref_path = ref_candidates[0]
                else:
                    logger.warning("No .wav file found in %s, skipping", voice_dir)
                    continue

            voice_id = voice_dir.name
            voice = Voice(
                id=voice_id,
                name=voice_id.replace("-", " ").replace("_", " ").title(),
                reference_path=ref_path,
                languages=list(XTTS_LANGUAGES),
            )
            self._voices[voice_id] = voice
            logger.info("Found voice: %s (%s)", voice_id, ref_path)

        logger.info("Scanned %d voice(s)", len(self._voices))
        return list(self._voices.values())

    def compute_conditioning(self, model) -> None:
        """Pre-compute gpt_cond_latent and speaker_embedding for all voices."""
        for voice in self._voices.values():
            try:
                logger.info("Computing conditioning for voice '%s'...", voice.id)
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=[str(voice.reference_path)],
                )
                voice.gpt_cond_latent = gpt_cond_latent
                voice.speaker_embedding = speaker_embedding
                logger.info("Conditioning computed for voice '%s'", voice.id)
            except Exception:
                logger.exception("Failed to compute conditioning for voice '%s'", voice.id)

    def get_voice(self, voice_id: str) -> Voice | None:
        return self._voices.get(voice_id)

    def list_voices_info(self) -> list[dict]:
        """Return voice info in the protocol format for the orchestrator."""
        return [
            {
                "id": v.id,
                "name": v.name,
                "languages": v.languages,
            }
            for v in self._voices.values()
        ]
