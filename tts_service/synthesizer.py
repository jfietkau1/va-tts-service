import logging
from collections.abc import Generator
from pathlib import Path

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from .voice_manager import Voice

logger = logging.getLogger(__name__)

XTTS_SAMPLE_RATE = 24000


class Synthesizer:
    """Wraps XTTS v2 model for streaming speech synthesis."""

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        use_deepspeed: bool = True,
    ):
        self._device = device
        self._model: Xtts | None = None
        self._model_path = model_path
        self._use_deepspeed = use_deepspeed

    def load(self) -> None:
        if self._model_path:
            checkpoint_dir = self._model_path
        else:
            from huggingface_hub import snapshot_download
            logger.info("Downloading XTTS v2 model from HuggingFace...")
            checkpoint_dir = snapshot_download("coqui/XTTS-v2")

        logger.info("Loading XTTS model from %s...", checkpoint_dir)

        config = XttsConfig()
        config.load_json(str(Path(checkpoint_dir) / "config.json"))

        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(
            config,
            checkpoint_dir=str(checkpoint_dir),
            use_deepspeed=self._use_deepspeed,
        )

        if self._device == "cuda" and torch.cuda.is_available():
            self._model.cuda()
            logger.info("XTTS model loaded on CUDA")
        else:
            logger.info("XTTS model loaded on CPU")

    @property
    def model(self) -> Xtts:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    def synthesize_stream(
        self,
        text: str,
        voice: Voice,
        language: str = "en",
    ) -> Generator[torch.Tensor, None, None]:
        """Stream audio chunks for the given text using the specified voice.

        Yields torch.Tensor chunks (float32, 24kHz mono).
        """
        if voice.gpt_cond_latent is None or voice.speaker_embedding is None:
            raise ValueError(f"Voice '{voice.id}' has no pre-computed conditioning")

        logger.debug("Synthesizing: '%s' (voice=%s, lang=%s)", text[:80], voice.id, language)

        chunks = self.model.inference_stream(
            text,
            language,
            voice.gpt_cond_latent,
            voice.speaker_embedding,
        )

        for chunk in chunks:
            yield chunk
