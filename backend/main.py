
from __future__ import annotations
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import asyncio
import logging
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect


from asr import Transcriber
from llm import (
    ClinicalSummary,
    SUMMARY_ERROR_MESSAGE,
    generate_clinical_summary,
)
from stream_processor import StreamProcessor
from vad import VoiceActivityDetector

# Application

app = FastAPI(
    title="Live Medical Scribe API",
    version="1.2.0",
)

logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=2)

# Configuration

SAMPLE_RATE = 16000

# Maximum duration of one complete WebSocket session.
MAX_SESSION_SECONDS = 30 * 60

# Maximum size of one incoming WebSocket audio message.
MAX_AUDIO_BYTES = 128 * 1024

# StreamProcessor configuration.
MAX_SPEECH_SEGMENT_SECONDS = 10.0
MIN_SILENCE_DURATION_SECONDS = 0.6
PRE_ROLL_DURATION_SECONDS = 0.2

# Heartbeat keeps the WebSocket active through proxies/load balancers.
HEARTBEAT_INTERVAL_SECONDS = 20

# Faster-Whisper configuration.
ASR_MODEL_SIZE = "base"
ASR_CPU_THREADS = 8
ASR_DEVICE = "cpu"
ASR_COMPUTE_TYPE = "int8"

# Server metrics

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

# Dependency checks
def _check_dependency(name: str, import_name: str) -> dict:
    """
    Check whether an optional/runtime dependency is available.
    """

    try:
        module = __import__(import_name)

        return {
            "name": name,
            "available": True,
            "version": getattr(module, "__version__", "unknown"),
        }

    except ImportError as exc:

        return {
            "name": name,
            "available": False,
            "error": str(exc),
        }


@app.get("/")
def read_root():
    return {
        "name": "Live Medical Scribe API",
        "status": "ok",
        "version": app.version,
        "features": [
            "VAD",
            "Multilingual Whisper ASR",
            "Automatic language detection",
            "Multilingual session switching",
            "Phrase-level streaming transcription",
            "Pre-roll audio buffering",
            "LLM clinical summary",
            "Pydantic clinical schema validation",
        ],
    }


@app.get("/health")
def health():
    """
    General health information.

    This endpoint reports whether the major ML dependencies
    can currently be imported.
    """

    dependencies = [
        _check_dependency("faster_whisper", "faster_whisper"),
        _check_dependency("silero_vad", "silero_vad"),
        _check_dependency("torch", "torch"),
        _check_dependency("numpy", "numpy"),
    ]

    all_ok = all(
        dependency["available"]
        for dependency in dependencies
    )

    return {
        "status": "healthy" if all_ok else "degraded",
        "uptime_seconds": round(
            time.monotonic() - SERVER_START_TIME,
            1,
        ),
        "started_at": SERVER_START_ISO,
        "dependencies": dependencies,
        "active_connections": len(_active_connections),
    }


@app.get("/health/live")
def health_live():
    """
    Liveness check.

    Used to confirm that the FastAPI process itself is running.
    """

    return {
        "status": "alive",
    }


@app.get("/health/ready")
def health_ready():
    """
    Readiness check.

    Returns HTTP 503 if required ML dependencies are unavailable.
    """

    dependencies = [
        _check_dependency("faster_whisper", "faster_whisper"),
        _check_dependency("silero_vad", "silero_vad"),
        _check_dependency("torch", "torch"),
    ]

    missing = [
        dependency["name"]
        for dependency in dependencies
        if not dependency["available"]
    ]

    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "missing": missing,
            },
        )

    return {
        "status": "ready",
        "dependencies": dependencies,
    }

# Status endpoint

@app.get("/status")
def status():
    """
    Runtime information about active WebSocket sessions
    and transcription performance.
    """

    now = time.monotonic()

    connections = [
        {
            "connection_id": connection_id,
            "connected_seconds": round(
                now - info["connected_at"],
                1,
            ),
            "stage": info.get("stage"),
            "segments_transcribed": info.get(
                "segments_transcribed",
                0,
            ),
            "current_language": info.get(
                "current_language"
            ),
            "last_error": info.get("last_error"),
        }
        for connection_id, info in _active_connections.items()
    ]

    latencies = _metrics["transcription_latencies_ms"]

    avg_latency = (
        round(sum(latencies) / len(latencies), 1)
        if latencies
        else None
    )

    return {
        "uptime_seconds": round(
            now - SERVER_START_TIME,
            1,
        ),
        "active_connections": len(_active_connections),
        "connections": connections,
        "metrics": {
            "total_connections_ever": _metrics[
                "total_connections"
            ],
            "total_errors": _metrics[
                "total_errors"
            ],
            "total_transcriptions": _metrics[
                "total_transcriptions"
            ],
            "avg_transcription_latency_ms": avg_latency,
        },
        "system": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "asr": {
            "model": ASR_MODEL_SIZE,
            "device": ASR_DEVICE,
            "compute_type": ASR_COMPUTE_TYPE,
            "cpu_threads": ASR_CPU_THREADS,
        },
    }


@app.get("/metrics")
def metrics():
    """
    Lightweight metrics endpoint.
    """

    latencies = _metrics["transcription_latencies_ms"]

    avg_latency = (
        round(sum(latencies) / len(latencies), 1)
        if latencies
        else None
    )

    return {
        "uptime_seconds": round(
            time.monotonic() - SERVER_START_TIME,
            1,
        ),
        "active_connections": len(_active_connections),
        "total_connections_ever": _metrics[
            "total_connections"
        ],
        "total_errors": _metrics[
            "total_errors"
        ],
        "total_transcriptions": _metrics[
            "total_transcriptions"
        ],
        "avg_transcription_latency_ms": avg_latency,
        "last_disconnect": _metrics[
            "last_disconnect"
        ],
    }

# WebSocket endpoint

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main real-time audio endpoint.

    Client sends:
        - binary PCM16 audio chunks
        - "stop" text message

    Server sends:
        - status
        - heartbeat
        - recording_language
        - transcript
        - summary
        - error
    """

    await websocket.accept()

    # Connection registration

    connection_id = (
        f"conn-{int(time.time() * 1000)}-"
        f"{id(websocket) % 10000}"
    )

    _metrics["total_connections"] += 1

    _active_connections[connection_id] = {
        "connected_at": time.monotonic(),
        "stage": "connecting",
        "segments_transcribed": 0,
        "current_language": None,
        "last_error": None,
    }

    session_started = time.monotonic()

    loop = asyncio.get_running_loop()

    # Runtime model objects
    
    detector: VoiceActivityDetector | None = None
    transcriber: Transcriber | None = None
    stream_processor: StreamProcessor | None = None

    # Complete transcript for final Gemini summary

    transcript_segments: list[dict] = []


    async def heartbeat_loop():
        """
        Send periodic heartbeat messages to keep the WebSocket
        """

        try:
            while True:

                await asyncio.sleep(
                    HEARTBEAT_INTERVAL_SECONDS
                )

                connection_info = _active_connections.get(
                    connection_id
                )

                if connection_info is None:
                    return

                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "uptime_seconds": round(
                            time.monotonic()
                            - connection_info["connected_at"],
                            1,
                        ),
                    }
                )

        except asyncio.CancelledError:
            # Expected when the WebSocket session finishes.
            return

        except Exception:
            # Socket may already be closed.
            return

    heartbeat_task = asyncio.create_task(
        heartbeat_loop()
    )

    # Model initialization
    
    async def initialize_models():
        nonlocal detector
        nonlocal transcriber
        nonlocal stream_processor

        if stream_processor is not None:
            return

        await websocket.send_json(
            {
                "type": "status",
                "stage": "initializing",
                "message": (
                    "Loading VAD and multilingual "
                    "Whisper..."
                ),
            }
        )

        _active_connections[connection_id][
            "stage"
        ] = "initializing"

        # Load Silero VAD

        detector = await loop.run_in_executor(
            executor,
            lambda: VoiceActivityDetector(
                threshold=0.5,
                hangover_frames=8,
            ),
        )
        # Load Faster-Whisper

        transcriber = await loop.run_in_executor(
            executor,
            lambda: Transcriber(
                model_size=ASR_MODEL_SIZE,
                device=ASR_DEVICE,
                compute_type=ASR_COMPUTE_TYPE,
                cpu_threads=ASR_CPU_THREADS,
                num_workers=1,
            ),
        )

        # Create streaming processor

        stream_processor = StreamProcessor(
            transcriber=transcriber,
            vad=detector,
            sample_rate=SAMPLE_RATE,
            max_speech_duration_s=(
                MAX_SPEECH_SEGMENT_SECONDS
            ),
            min_silence_duration_s=(
                MIN_SILENCE_DURATION_SECONDS
            ),
            pre_roll_duration_s=(
                PRE_ROLL_DURATION_SECONDS
            ),
        )

        await websocket.send_json(
            {
                "type": "status",
                "stage": "ready",
                "message": (
                    "VAD + multilingual Whisper ready"
                ),
            }
        )

        _active_connections[connection_id][
            "stage"
        ] = "ready"

    # Add transcript segment

    def add_transcript_segment(
        result: dict | None,
    ) -> dict | None:

        if not result:
            return None

        text = result.get("text", "").strip()

        if not text:
            return None

        segment = {
            "text": text,
            "language": result.get("language"),
            "language_name": result.get(
                "language_name"
            ),
            "language_confidence": result.get(
                "language_confidence"
            ),
        }

        transcript_segments.append(segment)

        connection_info = _active_connections.get(
            connection_id
        )

        if connection_info is not None:

            connection_info[
                "segments_transcribed"
            ] += 1

            connection_info[
                "current_language"
            ] = result.get("language_name")

        return segment

    # Send one transcription result

    async def process_transcription_result(
        result: dict | None,
    ):

        segment = add_transcript_segment(result)

        if not segment:
            return None

        await websocket.send_json(
            {
                "type": "transcript",
                **segment,
            }
        )

        await websocket.send_json(
            {
                "type": "recording_language",
                "language": segment.get(
                    "language"
                ),
                "language_name": segment.get(
                    "language_name"
                ),
                "confidence": segment.get(
                    "language_confidence"
                ),
            }
        )

        return segment

    # Final Gemini summary

    async def send_summary() -> None:

        if not transcript_segments:

            await websocket.send_json(
                {
                    "type": "summary",
                    "data": {
                        "error": "No transcript",
                    },
                }
            )

            return

        # Build language-aware transcript

        formatted_segments = []

        for segment in transcript_segments:

            language = (
                segment.get("language_name")
                or "Unknown"
            )

            text = (
                segment.get("text", "")
                .strip()
            )

            if text:

                formatted_segments.append(
                    f"[{language}]\n{text}"
                )

        transcript = "\n\n".join(
            formatted_segments
        ).strip()

        if not transcript:

            await websocket.send_json(
                {
                    "type": "summary",
                    "data": {
                        "error": "No transcript",
                    },
                }
            )

            return

        try:

            await websocket.send_json(
                {
                    "type": "status",
                    "stage": "summarizing",
                    "message": (
                        "Analyzing multilingual "
                        "transcript with LLM..."
                    ),
                }
            )

            _active_connections[
                connection_id
            ]["stage"] = "summarizing"

            summary: ClinicalSummary = (
                await loop.run_in_executor(
                    executor,
                    generate_clinical_summary,
                    transcript,
                )
            )

            # Pydantic v2 / v1 compatibility

            if hasattr(summary, "model_dump"):

                data = summary.model_dump()

            else:

                data = summary.dict()

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

            _active_connections[
                connection_id
            ]["last_error"] = (
                "summary_generation_failed"
            )

            _metrics["total_errors"] += 1

            await websocket.send_json(
                {
                    "type": "error",
                    "message": SUMMARY_ERROR_MESSAGE,
                }
            )
    # Main WebSocket receive loop
    try:

        while True:

            message = await websocket.receive()
            # WebSocket disconnect event
            if (
                message.get("type")
                == "websocket.disconnect"
            ):

                disconnect_code = message.get(
                    "code"
                )

                logger.info(
                    "WebSocket disconnect event received: "
                    "code=%s connection_id=%s",
                    disconnect_code,
                    connection_id,
                )

                _metrics[
                    "last_disconnect"
                ] = {
                    "connection_id": connection_id,
                    "code": disconnect_code,
                    "reason": (
                        "client_message_disconnect"
                    ),
                    "at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }

                break

            # CONTROL MESSAGES
            if message.get("text") is not None:

                control_message = message["text"]

                if (
                    not isinstance(
                        control_message,
                        str,
                    )
                    or len(control_message) > 32
                ):

                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": (
                                "Invalid control "
                                "message."
                            ),
                        }
                    )

                    continue

                # STOP

                if control_message == "stop":

                    if stream_processor is not None:

                        await websocket.send_json(
                            {
                                "type": "status",
                                "stage": "finishing",
                                "message": (
                                    "Processing remaining "
                                    "audio..."
                                ),
                            }
                        )

                        result = (
                            await loop.run_in_executor(
                                executor,
                                stream_processor.finish,
                            )
                        )

                        await process_transcription_result(
                            result
                        )

                    # Generate final clinical summary.
                    await send_summary()

                    # Close WebSocket after summary.
                    await websocket.close()

                    return
                # Unsupported control message

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Unsupported control "
                            "message."
                        ),
                    }
                )

                continue
            # AUDIO DATA
            data = message.get("bytes")

            if not data:
                continue
            # Audio packet size protection

            if len(data) > MAX_AUDIO_BYTES:

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Audio chunk is too large."
                        ),
                    }
                )

                continue

            # PCM16 validation

            if len(data) % 2:

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Audio chunk has an invalid "
                            "PCM format."
                        ),
                    }
                )

                continue

            # Session duration protection

            elapsed_session = (
                time.monotonic()
                - session_started
            )

            if (
                elapsed_session
                > MAX_SESSION_SECONDS
            ):

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Recording session "
                            "limit reached."
                        ),
                    }
                )

                await websocket.close(
                    code=1008
                )

                return

            # Lazy model initialization
            await initialize_models()
            if stream_processor is None:
                raise RuntimeError(
                    "StreamProcessor failed to initialize."
                )

            # Convert PCM16 -> float32
            audio_chunk = (
                np.frombuffer(
                    data,
                    dtype=np.int16,
                ).astype(
                    np.float32
                )
                / 32768.0
            )

            if audio_chunk.size == 0:
                continue

            _active_connections[
                connection_id
            ]["stage"] = "recording"

            t0 = time.monotonic()

            result = await loop.run_in_executor(
                executor,
                stream_processor.process_chunk,
                audio_chunk,
            )

            if result is not None:

                elapsed_ms = (
                    time.monotonic() - t0
                ) * 1000

                _metrics[
                    "transcription_latencies_ms"
                ].append(elapsed_ms)

                # Keep only recent latency samples.
                _metrics[
                    "transcription_latencies_ms"
                ] = _metrics[
                    "transcription_latencies_ms"
                ][-200:]

                _metrics[
                    "total_transcriptions"
                ] += 1

                await process_transcription_result(
                    result
                )

    # Expected WebSocket disconnect
 
    except WebSocketDisconnect as exc:

        logger.info(
            "WebSocket client disconnected: "
            "code=%s reason=%s connection_id=%s",
            getattr(exc, "code", None),
            getattr(exc, "reason", None),
            connection_id,
        )

        _metrics[
            "last_disconnect"
        ] = {
            "connection_id": connection_id,
            "code": getattr(exc, "code", None),
            "reason": (
                getattr(exc, "reason", None)
                or "websocket_disconnect_exception"
            ),
            "at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    # Unexpected processing error

    except Exception:

        logger.exception(
            "WebSocket processing failed"
        )

        connection_info = _active_connections.get(
            connection_id
        )

        if connection_info is not None:

            connection_info[
                "last_error"
            ] = "processing_failed"

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
            # Socket may already be closed.
            pass


    finally:

        heartbeat_task.cancel()

        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

        _active_connections.pop(
            connection_id,
            None,
        )