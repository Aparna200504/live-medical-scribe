# asr.py — back to no prompt biasing, keeping everything else from before
from __future__ import annotations

import numpy as np


LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "gu": "Gujarati",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu",
}

LANGUAGE_CONFIDENCE_FLOOR = 0.6


class Transcriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 6,
        num_workers: int = 1,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Speech recognition is unavailable. "
                "Install the dependencies from requirements.txt."
            ) from exc

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

        self._locked_language: str | None = None
        self._locked_confidence: float = 0.0

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        if audio.size == 0:
            return {"text": "", "language": None, "language_name": None}

        min_samples = int(0.6 * sample_rate)
        if audio.size < min_samples:
            return {"text": "", "language": None, "language_name": None}

        segments, info = self.model.transcribe(
            audio,
            beam_size=2,
            language=None,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0,
        )

        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        detected_language = getattr(info, "language", None)
        confidence = getattr(info, "language_probability", 0.0) or 0.0
        language = self._resolve_language(detected_language, confidence)

        return {
            "text": text,
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language.upper() if language else "Unknown"),
            "language_confidence": confidence,
        }

    def _resolve_language(self, detected: str | None, confidence: float) -> str | None:
        if detected is None:
            return self._locked_language
        if confidence >= LANGUAGE_CONFIDENCE_FLOOR or self._locked_language is None:
            self._locked_language = detected
            self._locked_confidence = confidence
            return detected
        return self._locked_language