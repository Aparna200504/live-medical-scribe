import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from asr import Transcriber
from llm import ClinicalSummary, SUMMARY_ERROR_MESSAGE, generate_clinical_summary
from vad import VoiceActivityDetector

app = FastAPI(title="Live Medical Scribe API", version="1.0.0")
logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=2)

MAX_SESSION_SECONDS = 30 * 60
MAX_AUDIO_BYTES = 128 * 1024
SILENCE_THRESHOLD_SECONDS = 0.6
MAX_SPEECH_SEGMENT_SECONDS = 8

@app.get("/")
def read_root():
    return {"name": "Live Medical Scribe API", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy", "services": ["VAD", "Whisper ASR", "LLM"]}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    detector = None
    transcriber = None
    speech_buffer: list[np.ndarray] = []
    full_transcript: list[str] = []
    silence_duration = 0.0
    session_started = time.monotonic()
    loop = asyncio.get_running_loop()

    async def initialize_models():
        nonlocal detector, transcriber
        if detector is None:
            await websocket.send_json({"type": "status", "stage": "initializing", "message": "Loading VAD and Whisper..."})
            detector = await loop.run_in_executor(executor, VoiceActivityDetector)
            transcriber = await loop.run_in_executor(executor, Transcriber)
            await websocket.send_json({"type": "status", "stage": "ready", "message": "VAD + Whisper ready"})

    async def transcribe_buffer() -> str:
        nonlocal speech_buffer, silence_duration
        if not speech_buffer or transcriber is None:
            return ""
        segment_audio = np.concatenate(speech_buffer)
        speech_buffer = []
        silence_duration = 0.0
        await websocket.send_json({"type": "status", "stage": "transcribing", "message": "Whisper is transcribing speech..."})
        text = await loop.run_in_executor(executor, transcriber.transcribe, segment_audio)
        await websocket.send_json({"type": "status", "stage": "recording", "message": "Listening — speech detected by VAD"})
        return text

    async def send_summary() -> None:
        text = " ".join(full_transcript).strip()
        if not text:
            await websocket.send_json({"type": "summary", "data": {"error": "No transcript"}})
            return
        try:
            await websocket.send_json({"type": "status", "stage": "summarizing", "message": "Analyzing transcript with LLM..."})
            summary: ClinicalSummary = await loop.run_in_executor(executor, generate_clinical_summary, text)
            data = summary.model_dump() if hasattr(summary, "model_dump") else summary.dict()
            await websocket.send_json({"type": "summary", "data": data})
        except Exception as exc:
            logger.exception("Clinical summary generation failed: %s", exc)
            await websocket.send_json({"type": "error", "message": SUMMARY_ERROR_MESSAGE})

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                control_message = message["text"]
                if not isinstance(control_message, str) or len(control_message) > 32:
                    await websocket.send_json({"type": "error", "message": "Invalid control message."})
                    continue
                if control_message == "stop":
                    text = await transcribe_buffer()
                    if text:
                        full_transcript.append(text)
                        await websocket.send_json({"type": "transcript", "text": text})
                    await send_summary()
                    await websocket.close()
                    return
                await websocket.send_json({"type": "error", "message": "Unsupported control message."})
                continue

            data = message.get("bytes")
            if not data:
                continue
            if len(data) > MAX_AUDIO_BYTES:
                await websocket.send_json({"type": "error", "message": "Audio chunk is too large."})
                continue
            if len(data) % 2:
                await websocket.send_json({"type": "error", "message": "Audio chunk has an invalid PCM format."})
                continue
            if time.monotonic() - session_started > MAX_SESSION_SECONDS:
                await websocket.send_json({"type": "error", "message": "Recording session limit reached."})
                await websocket.close(code=1008)
                return

            await initialize_models()

            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            is_speech = await loop.run_in_executor(executor, detector.is_speech, audio_chunk)

            if is_speech:
                speech_buffer.append(audio_chunk)
                silence_duration = 0.0
            else:
                silence_duration += len(audio_chunk) / 16000.0
                if silence_duration >= SILENCE_THRESHOLD_SECONDS:
                    text = await transcribe_buffer()
                    if text:
                        full_transcript.append(text)
                        await websocket.send_json({"type": "transcript", "text": text})

            buffered_seconds = sum(len(chunk) for chunk in speech_buffer) / 16000.0
            if buffered_seconds >= MAX_SPEECH_SEGMENT_SECONDS:
                text = await transcribe_buffer()
                if text:
                    full_transcript.append(text)
                    await websocket.send_json({"type": "transcript", "text": text})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket processing failed")
        try:
            await websocket.send_json({"type": "error", "message": "Audio processing failed. Check the backend console."})
        except Exception:
            pass
