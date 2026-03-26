import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from .audio_player import AudioPlayer
from .synthesizer import Synthesizer
from .voice_manager import VoiceManager

logger = logging.getLogger(__name__)


@dataclass
class SpeakRequest:
    text: str
    voice: str
    language: str


class SpeechQueue:
    """Async queue that processes speak requests sequentially.

    Each request is synthesized via XTTS streaming and played through
    the audio player. Supports interruption via stop().
    """

    def __init__(
        self,
        synthesizer: Synthesizer,
        voice_manager: VoiceManager,
        audio_player: AudioPlayer,
        status_callback: asyncio.coroutines = None,
    ):
        self._synthesizer = synthesizer
        self._voice_manager = voice_manager
        self._player = audio_player
        self._status_callback = status_callback
        self._queue: asyncio.Queue[SpeakRequest | None] = asyncio.Queue()
        self._processing = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._processing = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Speech queue started")

    async def stop_processing(self) -> None:
        self._processing = False
        await self._queue.put(None)
        if self._task:
            await self._task

    async def enqueue(self, request: SpeakRequest) -> None:
        await self._queue.put(request)

    def interrupt(self) -> None:
        """Stop current playback and clear the queue."""
        self._player.stop()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _emit_status(self, status: dict) -> None:
        if self._status_callback:
            await self._status_callback(status)

    async def _process_loop(self) -> None:
        while self._processing:
            request = await self._queue.get()

            if request is None:
                self._queue.task_done()
                break

            await self._synthesize_and_play(request)
            self._queue.task_done()

            if self._queue.empty():
                await self._emit_status({"type": "idle"})

    async def _synthesize_and_play(self, request: SpeakRequest) -> None:
        voice = self._voice_manager.get_voice(request.voice)
        if not voice:
            logger.warning("Voice '%s' not found, falling back to first available", request.voice)
            voices = list(self._voice_manager.voices.values())
            if not voices:
                logger.error("No voices available")
                return
            voice = voices[0]

        await self._emit_status({"type": "speaking", "text": request.text})

        loop = asyncio.get_running_loop()
        try:
            chunks = await loop.run_in_executor(
                None,
                lambda: list(
                    self._synthesizer.synthesize_stream(
                        request.text,
                        voice,
                        request.language,
                    )
                ),
            )

            for chunk in chunks:
                audio = chunk.squeeze().cpu().numpy()
                self._player.enqueue(audio)

            self._player.wait_until_done()
        except Exception:
            logger.exception("Synthesis error for text: %s", request.text[:80])
