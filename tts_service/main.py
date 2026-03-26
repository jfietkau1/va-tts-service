import asyncio
import logging
import signal

from .config import settings
from .synthesizer import Synthesizer
from .voice_manager import VoiceManager
from .audio_player import AudioPlayer
from .speech_queue import SpeechQueue
from .ws_server import WsServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Initializing TTS service...")

    voice_manager = VoiceManager(voices_dir=settings.voices_dir)
    voices = voice_manager.scan()

    if not voices:
        logger.error(
            "No voices found in %s. Add a subdirectory with a reference.wav file.",
            settings.voices_dir,
        )
        return

    synthesizer = Synthesizer(
        model_path=settings.xtts_model_path,
        device=settings.xtts_device,
        use_deepspeed=settings.xtts_use_deepspeed,
    )
    synthesizer.load()
    voice_manager.compute_conditioning(synthesizer.model)

    audio_player = AudioPlayer(device_index=settings.audio_device_index)
    audio_player.start()

    ws_server = None

    async def status_callback(status: dict) -> None:
        if ws_server:
            await ws_server.broadcast(status)

    speech_queue = SpeechQueue(
        synthesizer=synthesizer,
        voice_manager=voice_manager,
        audio_player=audio_player,
        status_callback=status_callback,
    )

    ws_server = WsServer(
        host=settings.ws_host,
        port=settings.ws_port,
        speech_queue=speech_queue,
        voice_manager=voice_manager,
        default_voice=settings.default_voice,
        default_language=settings.default_language,
    )

    await ws_server.start()
    await speech_queue.start()

    logger.info(
        "TTS service running with %d voice(s). Default: %s (%s)",
        len(voices),
        settings.default_voice,
        settings.default_language,
    )

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_signal() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    await shutdown_event.wait()

    logger.info("Shutting down...")
    await speech_queue.stop_processing()
    audio_player.shutdown()
    await ws_server.stop()
    logger.info("TTS service stopped")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
