import websockets
import json
import asyncio
import base64
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed chunk size of 100ms
ASSEMBLY_CHUNK_SIZE_MS = 100  # Chunk Size in ms, fixed at 100ms for consistency

class TranscriptManager:
    def __init__(self):
        self.partial_transcripts: Dict[str, str] = {"inbound": "", "outbound": ""}
        self.final_transcripts: List[Tuple[str, str, int, int]] = []  # List of (track, text, start_ts, end_ts) tuples

    def update_partial(self, track: str, text: str):
        self.partial_transcripts[track] = text

    def update_final(self, track: str, text: str, start_ts: int, end_ts: int):
        self.final_transcripts.append((track, text, start_ts, end_ts))
        self.partial_transcripts[track] = ""

    def get_display_lines(self) -> List[str]:
        lines = []
        for track, transcript, start_ts, end_ts in self.final_transcripts:
            lines.append(f"{track.capitalize()}: {transcript} [{start_ts} - {end_ts}]")
        for track, partial in self.partial_transcripts.items():
            if partial:
                lines.append(f"{track.capitalize()}: {partial}")
        return lines

class AssemblyAITranscriber:
    def __init__(self, api_key, sample_rate=8000):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.ws_url = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={sample_rate}&encoding=pcm_mulaw"
        self.transcript_manager = TranscriptManager()
        self.chunk_size_samples = int(self.sample_rate * ASSEMBLY_CHUNK_SIZE_MS / 1000)

    async def transcribe(self, audio_queue, track_type):
        logger.info(f"Starting transcription for {track_type} track")
        async with websockets.connect(self.ws_url, extra_headers={"Authorization": self.api_key}) as websocket:
            await websocket.recv()  # Receive the ready message
            logger.info(f"WebSocket connection established for {track_type} track")

            async def send_audio():
                buffer = b""
                while True:
                    try:
                        audio_data = await audio_queue.get()
                        if audio_data is None:  # None is our signal to stop
                            logger.info(f"Received stop signal for {track_type} track")
                            break
                        buffer += audio_data
                        while len(buffer) >= self.chunk_size_samples:
                            chunk = buffer[:self.chunk_size_samples]
                            buffer = buffer[self.chunk_size_samples:]
                            # Encode the binary audio data to base64
                            audio_base64 = base64.b64encode(chunk).decode('utf-8')
                            await websocket.send(json.dumps({"audio_data": audio_base64}))
                    except asyncio.CancelledError:
                        logger.info(f"Send audio task cancelled for {track_type} track")
                        break

                # Handle any remaining audio in the buffer
                if buffer:
                    # Calculate how much silence to add
                    silence_needed = self.chunk_size_samples - len(buffer)
                    # Add silence (zeros) to the buffer
                    buffer += b'\x00' * silence_needed
                    # Encode and send the padded chunk
                    audio_base64 = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send(json.dumps({"audio_data": audio_base64}))
                    logger.info(f"Sent final padded chunk for {track_type} track")

                await websocket.send(json.dumps({"terminate_session": True}))
                logger.info(f"Sent terminate session message for {track_type} track")

            async def receive_transcripts():
                while True:
                    try:
                        message = await websocket.recv()
                        response = json.loads(message)
                        logger.debug(f"Received message for {track_type} track: {response}")
                        if response.get("message_type") == "PartialTranscript":
                            if response.get("text"):
                                self.transcript_manager.update_partial(track_type, response["text"])
                                await self.print_transcripts()
                        elif response.get("message_type") == "FinalTranscript":
                            if response.get("text"):
                                start_ts = response.get("audio_start")
                                end_ts = response.get("audio_end")
                                self.transcript_manager.update_final(track_type, response["text"], start_ts, end_ts)
                                await self.print_transcripts()
                        elif response.get("message_type") == "SessionTerminated":
                            logger.info(f"Session terminated for {track_type} track")
                            break
                    except asyncio.CancelledError:
                        logger.info(f"Receive transcripts task cancelled for {track_type} track")
                        break

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_transcripts())

            await asyncio.gather(send_task, receive_task)

    async def print_transcripts(self):
        # Clear the console (works on most terminals)
        print("\033[H\033[J", end="")
        
        for line in self.transcript_manager.get_display_lines():
            print(line)

async def run_transcription(api_key, inbound_queue, outbound_queue):
    transcriber = AssemblyAITranscriber(api_key)
    inbound_task = asyncio.create_task(transcriber.transcribe(inbound_queue, "inbound"))
    outbound_task = asyncio.create_task(transcriber.transcribe(outbound_queue, "outbound"))
    await asyncio.gather(inbound_task, outbound_task)