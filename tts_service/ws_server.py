import asyncio
import json
import logging

import websockets
from websockets.asyncio.server import Server, ServerConnection

from .audio_player import AudioPlayer
from .sound_manager import SoundManager
from .speech_queue import SpeakRequest, SpeechQueue
from .voice_manager import VoiceManager

logger = logging.getLogger(__name__)


class WsServer:
    """WebSocket server that receives TTS commands from the orchestrator.

    Handles speak, stop, config, and listVoices commands.
    Sends status messages (speaking, idle, stopped, voices) back.
    """

    def __init__(
        self,
        host: str,
        port: int,
        speech_queue: SpeechQueue,
        voice_manager: VoiceManager,
        sound_manager: SoundManager,
        audio_player: AudioPlayer,
        default_voice: str,
        default_language: str,
    ):
        self._host = host
        self._port = port
        self._speech_queue = speech_queue
        self._voice_manager = voice_manager
        self._sound_manager = sound_manager
        self._audio_player = audio_player
        self._default_voice = default_voice
        self._default_language = default_language
        self._clients: set[ServerConnection] = set()
        self._server: Server | None = None

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
        )
        logger.info("TTS WebSocket server listening on ws://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("TTS WebSocket server stopped")

    async def broadcast(self, message: dict) -> None:
        """Send a status message to all connected clients."""
        if not self._clients:
            return

        data = json.dumps(message)
        disconnected = set()
        for client in self._clients:
            try:
                await client.send(data)
            except websockets.ConnectionClosed:
                disconnected.add(client)
        self._clients -= disconnected

    async def _handler(self, websocket: ServerConnection) -> None:
        self._clients.add(websocket)
        remote = websocket.remote_address
        logger.info("Client connected: %s", remote)

        try:
            async for raw in websocket:
                try:
                    command = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from client")
                    continue

                await self._handle_command(command, websocket)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected: %s", remote)

    async def _handle_command(self, command: dict, websocket: ServerConnection) -> None:
        cmd_type = command.get("type")

        if cmd_type == "speak":
            request = SpeakRequest(
                text=command.get("text", ""),
                voice=command.get("voice", self._default_voice),
                language=command.get("language", self._default_language),
            )
            if request.text:
                await self._speech_queue.enqueue(request)
            else:
                logger.warning("Empty speak command received")

        elif cmd_type == "stop":
            self._speech_queue.interrupt()
            await self.broadcast({"type": "stopped"})

        elif cmd_type == "config":
            new_voice = command.get("voice")
            new_lang = command.get("language")
            if new_voice:
                self._default_voice = new_voice
            if new_lang:
                self._default_language = new_lang
            logger.info(
                "Config updated: voice=%s, language=%s",
                self._default_voice,
                self._default_language,
            )

        elif cmd_type == "listVoices":
            voices = self._voice_manager.list_voices_info()
            await websocket.send(json.dumps({"type": "voices", "voices": voices}))

        elif cmd_type == "playSound":
            sound_id = command.get("sound", "")
            audio = self._sound_manager.get(sound_id)
            if audio is not None:
                self._audio_player.enqueue(audio)
                logger.debug("Playing sound: %s", sound_id)
            else:
                logger.warning("Sound not found: %s", sound_id)

        else:
            logger.warning("Unknown command type: %s", cmd_type)
