import logging
import queue
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

XTTS_SAMPLE_RATE = 24000
BLOCK_SIZE = 1024


class AudioPlayer:
    """Plays audio chunks through the speaker using sounddevice.OutputStream.

    Supports immediate stop (clears the queue and halts playback).
    Runs the output stream in a background thread for non-blocking operation.
    """

    def __init__(self, device_index: int = -1):
        self._device = device_index if device_index >= 0 else None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._thread: threading.Thread | None = None
        self._playing = False
        self._stopped = threading.Event()

    def start(self) -> None:
        """Start the audio output stream."""
        if self._thread and self._thread.is_alive():
            return

        self._playing = True
        self._stopped.clear()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        logger.info("Audio player started")

    def enqueue(self, chunk: np.ndarray) -> None:
        """Add an audio chunk to the playback queue.

        Chunk should be float32 numpy array at 24kHz.
        """
        if self._playing:
            self._queue.put(chunk)

    def stop(self) -> None:
        """Stop playback immediately and clear the queue."""
        self._clear_queue()
        self._stopped.set()
        logger.debug("Audio playback stopped")

    def shutdown(self) -> None:
        """Shut down the player entirely."""
        self._playing = False
        self._clear_queue()
        self._queue.put(None)
        self._stopped.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Audio player shut down")

    def wait_until_done(self) -> bool:
        """Block until the queue is empty or stop is called.

        Returns True if playback completed, False if stopped.
        """
        self._queue.join()
        return not self._stopped.is_set()

    def _clear_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _playback_loop(self) -> None:
        while self._playing:
            self._stopped.clear()

            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None:
                self._queue.task_done()
                break

            self._play_chunk(chunk)
            self._queue.task_done()

    def _play_chunk(self, audio: np.ndarray) -> None:
        if audio.ndim > 1:
            audio = audio.squeeze()

        audio_float = audio.astype(np.float32)
        if audio_float.max() > 1.0 or audio_float.min() < -1.0:
            audio_float = np.clip(audio_float, -1.0, 1.0)

        try:
            sd.play(audio_float, samplerate=XTTS_SAMPLE_RATE, device=self._device)
            while sd.get_stream().active and not self._stopped.is_set():
                self._stopped.wait(timeout=0.05)
            if self._stopped.is_set():
                sd.stop()
        except Exception:
            logger.exception("Error during audio playback")
