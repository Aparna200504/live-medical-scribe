import os
import platform
import sys
import asyncio
import logging
import time
from datetime import datetime, timezone
from fastapi import HTTPException
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from asr import Transcriber
from llm import ClinicalSummary, SUMMARY_ERROR_MESSAGE, generate_clinical_summary
from vad import VoiceActivityDetector

# main.py — at the very top, before importing asr/vad/torch

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")


app = FastAPI(
    title="Live Medical Scribe API",
    version="1.1.0",
)

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=2)

MAX_SESSION_SECONDS = 30 * 60
MAX_AUDIO_BYTES = 128 * 1024
SILENCE_THRESHOLD_SECONDS = 1.0
MAX_SPEECH_SEGMENT_SECONDS = 10
MIN_SPEECH_SEGMENT_SECONDS = 0.8
HEARTBEAT_INTERVAL_SECONDS = 20   # keep well under typical proxy/idle timeouts
MIN_CONNECTION_SECONDS = 5 * 60   # documented floor — heartbeat is what enforces this in practice

SERVER_START_TIME = time.monotonic()
SERVER_START_ISO = datetime.now(timezone.utc).isoformat()

_active_connections: dict[str, dict] = {}
_metrics = {
    "total_connections": 0,
    "total_errors": 0,
    "total_transcriptions": 0,
    "transcription_latencies_ms": [],
    "last_disconnect": None,
}


def _check_dependency(name: str, import_name: str) -> dict:
    try:
        module = __import__(import_name)
        return {"name": name, "available": True, "version": getattr(module, "__version__", "unknown")}
    except ImportError as exc:
        return {"name": name, "available": False, "error": str(exc)}


@app.get("/")
def read_root():
    return {
        "name": "Live Medical Scribe API",
        "status": "ok",
        "features": [
            "VAD",
            "Multilingual Whisper ASR",
            "Automatic language detection",
            "Multilingual session switching",
            "LLM clinical summary",
        ],
    }


@app.get("/health")
def health():
    dependencies = [
        _check_dependency("faster_whisper", "faster_whisper"),
        _check_dependency("silero_vad", "silero_vad"),
        _check_dependency("torch", "torch"),
        _check_dependency("numpy", "numpy"),
    ]
    all_ok = all(d["available"] for d in dependencies)
    return {
        "status": "healthy" if all_ok else "degraded",
        "uptime_seconds": round(time.monotonic() - SERVER_START_TIME, 1),
        "started_at": SERVER_START_ISO,
        "dependencies": dependencies,
        "active_connections": len(_active_connections),
    }


@app.get("/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready():
    dependencies = [
        _check_dependency("faster_whisper", "faster_whisper"),
        _check_dependency("silero_vad", "silero_vad"),
        _check_dependency("torch", "torch"),
    ]
    missing = [d["name"] for d in dependencies if not d["available"]]
    if missing:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "missing": missing})
    return {"status": "ready", "dependencies": dependencies}


@app.get("/status")
def status():
    now = time.monotonic()
    connections = [
        {
            "connection_id": conn_id,
            "connected_seconds": round(now - info["connected_at"], 1),
            "stage": info.get("stage"),
            "segments_transcribed": info.get("segments_transcribed", 0),
            "current_language": info.get("current_language"),
            "last_error": info.get("last_error"),
        }
        for conn_id, info in _active_connections.items()
    ]
    latencies = _metrics["transcription_latencies_ms"]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None
    return {
        "uptime_seconds": round(now - SERVER_START_TIME, 1),
        "active_connections": len(_active_connections),
        "connections": connections,
        "metrics": {
            "total_connections_ever": _metrics["total_connections"],
            "total_errors": _metrics["total_errors"],
            "total_transcriptions": _metrics["total_transcriptions"],
            "avg_transcription_latency_ms": avg_latency,
        },
        "system": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }


@app.get("/metrics")
def metrics():
    latencies = _metrics["transcription_latencies_ms"]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None
    return {
        "uptime_seconds": round(time.monotonic() - SERVER_START_TIME, 1),
        "active_connections": len(_active_connections),
        "total_connections_ever": _metrics["total_connections"],
        "total_errors": _metrics["total_errors"],
        "total_transcriptions": _metrics["total_transcriptions"],
        "avg_transcription_latency_ms": avg_latency,
        "last_disconnect": _metrics["last_disconnect"]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    connection_id = f"conn-{int(time.time() * 1000)}-{id(websocket) % 10000}"
    _metrics["total_connections"] += 1
    _active_connections[connection_id] = {
        "connected_at": time.monotonic(),
        "stage": "connecting",
        "segments_transcribed": 0,
        "current_language": None,
        "last_error": None,
    }

    async def heartbeat_loop():
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                await websocket.send_json({
                    "type": "heartbeat",
                    "uptime_seconds": round(
                        time.monotonic() - _active_connections[connection_id]["connected_at"], 1
                    ),
                })
        except Exception:
            pass  # socket closed — loop just stops, cleanup happens in finally











    heartbeat_task = asyncio.create_task(heartbeat_loop())

    detector = None
    transcriber = None

    speech_buffer: list[np.ndarray] = []

    # Store structured transcript segments.
    # Each segment remembers the language detected by Whisper.
    transcript_segments: list[dict] = []

    silence_duration = 0.0
    session_started = time.monotonic()

    loop = asyncio.get_running_loop()

    async def initialize_models():
        nonlocal detector, transcriber

        if detector is None:
            await websocket.send_json(
                {
                    "type": "status",
                    "stage": "initializing",
                    "message": "Loading VAD and multilingual Whisper...",
                }
            )
            _active_connections[connection_id]["stage"] = "initializing"

            detector = await loop.run_in_executor(
                executor,
                VoiceActivityDetector,
            )

            transcriber = await loop.run_in_executor(
                executor,
                Transcriber,
            )

            await websocket.send_json(
                {
                    "type": "status",
                    "stage": "ready",
                    "message": "VAD + multilingual Whisper ready",
                }
            )
            _active_connections[connection_id]["stage"] = "ready"

    async def transcribe_buffer() -> dict:
        nonlocal speech_buffer, silence_duration

        if not speech_buffer or transcriber is None:
            return {
                "text": "",
                "language": None,
                "language_name": None,
            }

        segment_audio = np.concatenate(speech_buffer)

        speech_buffer = []
        silence_duration = 0.0

        await websocket.send_json(
            {
                "type": "status",
                "stage": "transcribing",
                "message": "Detecting language and transcribing speech...",
            }
        )
        _active_connections[connection_id]["stage"] = "transcribing"

        t0 = time.monotonic()
        result = await loop.run_in_executor(
            executor,
            transcriber.transcribe,
            segment_audio,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        _metrics["transcription_latencies_ms"].append(elapsed_ms)
        _metrics["transcription_latencies_ms"] = _metrics["transcription_latencies_ms"][-200:]
        _metrics["total_transcriptions"] += 1

        await websocket.send_json(
            {
                "type": "recording_language",
                "language": result.get("language"),
                "language_name": result.get("language_name"),
            }
        )

        await websocket.send_json(
            {
                "type": "status",
                "stage": "recording",
                "message": (
                    f"Listening — "
                    f"{result.get('language_name') or 'language detected'}"
                ),
            }
        )
        _active_connections[connection_id]["stage"] = "recording"

        return result

    async def send_summary() -> None:
        if not transcript_segments:
            await websocket.send_json(
                {
                    "type": "summary",
                    "data": {
                        "error": "No transcript"
                    },
                }
            )
            return

        # Build a language-aware transcript for the LLM.
        formatted_segments = []

        for segment in transcript_segments:
            language = segment.get("language_name") or "Unknown"
            text = segment.get("text", "").strip()

            if text:
                formatted_segments.append(
                    f"[{language}]\n{text}"
                )

        transcript = "\n\n".join(formatted_segments).strip()

        if not transcript:
            await websocket.send_json(
                {
                    "type": "summary",
                    "data": {
                        "error": "No transcript"
                    },
                }
            )
            return

        try:
            await websocket.send_json(
                {
                    "type": "status",
                    "stage": "summarizing",
                    "message": "Analyzing multilingual transcript with LLM...",
                }
            )
            _active_connections[connection_id]["stage"] = "summarizing"

            summary: ClinicalSummary = await loop.run_in_executor(
                executor,
                generate_clinical_summary,
                transcript,
            )

            data = (
                summary.model_dump()
                if hasattr(summary, "model_dump")
                else summary.dict()
            )

            await websocket.send_json(
                {
                    "type": "summary",
                    "data": data,
                }
            )

        except Exception as exc:
            logger.exception(
                "Clinical summary generation failed: %s",
                exc,
            )

            await websocket.send_json(
                {
                    "type": "error",
                    "message": SUMMARY_ERROR_MESSAGE,
                }
            )

    def add_transcript_segment(result: dict):
        text = result.get("text", "").strip()

        if not text:
            return None

        segment = {
            "text": text,
            "language": result.get("language"),
            "language_name": result.get("language_name"),
        }

        transcript_segments.append(segment)
        _active_connections[connection_id]["segments_transcribed"] += 1
        _active_connections[connection_id]["current_language"] = result.get("language_name")

        return segment

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                disconnect_code = message.get("code")
                logger.info(
                    "WebSocket disconnect event received: code=%s connection_id=%s",
                    disconnect_code, connection_id,
                )
                _metrics["last_disconnect"] = {
                    "connection_id": connection_id,
                    "code": disconnect_code,
                    "reason": "client_message_disconnect",
                    "at": datetime.now(timezone.utc).isoformat(),
                }
                break

            # ---------------------------------------------------------
            # CONTROL MESSAGES
            # ---------------------------------------------------------

            if message.get("text") is not None:
                control_message = message["text"]

                if (
                    not isinstance(control_message, str)
                    or len(control_message) > 32
                ):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Invalid control message.",
                        }
                    )
                    continue

                if control_message == "stop":

                    result = await transcribe_buffer()

                    segment = add_transcript_segment(result)

                    if segment:
                        await websocket.send_json(
                            {
                                "type": "transcript",
                                **segment,
                            }
                        )

                    await send_summary()

                    await websocket.close()

                    return

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Unsupported control message.",
                    }
                )

                continue

            # ---------------------------------------------------------
            # AUDIO DATA
            # ---------------------------------------------------------

            data = message.get("bytes")

            if not data:
                continue

            if len(data) > MAX_AUDIO_BYTES:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Audio chunk is too large.",
                    }
                )
                continue

            if len(data) % 2:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Audio chunk has an invalid PCM format.",
                    }
                )
                continue

            if (
                time.monotonic() - session_started
                > MAX_SESSION_SECONDS
            ):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Recording session limit reached.",
                    }
                )

                await websocket.close(code=1008)

                return

            await initialize_models()

            audio_chunk = (
                np.frombuffer(
                    data,
                    dtype=np.int16,
                ).astype(np.float32)
                / 32768.0
            )

            is_speech = await loop.run_in_executor(
                executor,
                detector.is_speech,
                audio_chunk,
            )

            if is_speech:

                speech_buffer.append(audio_chunk)

                silence_duration = 0.0

            else:

                silence_duration += (
                    len(audio_chunk) / 16000.0
                )

                if (
                    silence_duration >= SILENCE_THRESHOLD_SECONDS
                    and buffered_seconds >= MIN_SPEECH_SEGMENT_SECONDS
                ):
                    result = await transcribe_buffer()

                    segment = add_transcript_segment(result)

                    if segment:

                        await websocket.send_json(
                            {
                                "type": "transcript",
                                **segment,
                            }
                        )

            # ---------------------------------------------------------
            # MAX SPEECH SEGMENT LIMIT
            # ---------------------------------------------------------

            buffered_seconds = (
                sum(len(chunk) for chunk in speech_buffer)
                / 16000.0
            )

            if (
                buffered_seconds
                >= MAX_SPEECH_SEGMENT_SECONDS
            ):

                result = await transcribe_buffer()

                segment = add_transcript_segment(result)

                if segment:

                    await websocket.send_json(
                        {
                            "type": "transcript",
                            **segment,
                        }
                    )
    except WebSocketDisconnect as exc:
        logger.info(
            "WebSocket client disconnected: code=%s reason=%s connection_id=%s",
            getattr(exc, "code", None), getattr(exc, "reason", None), connection_id,
        )
        _metrics["last_disconnect"] = {
            "connection_id": connection_id,
            "code": getattr(exc, "code", None),
            "reason": getattr(exc, "reason", None) or "websocket_disconnect_exception",
            "at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception:

        logger.exception(
            "WebSocket processing failed"
        )
        _active_connections[connection_id]["last_error"] = "processing_failed"
        _metrics["total_errors"] += 1

        try:

            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        "Audio processing failed. "
                        "Check the backend console."
                    ),
                }
            )

        except Exception:
            pass
    finally:
        heartbeat_task.cancel()
        _active_connections.pop(connection_id, None)