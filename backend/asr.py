from __future__ import annotations

import numpy as np


class Transcriber:
    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Speech recognition is unavailable. Install the dependencies from requirements.txt."
            ) from exc

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        del sample_rate
        if audio.size == 0:
            return ""

        segments, _ = self.model.transcribe(audio, language="en", beam_size=1)
        return " ".join(segment.text.strip() for segment in segments).strip()