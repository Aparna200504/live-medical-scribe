# vad.py
from __future__ import annotations

import numpy as np


class VoiceActivityDetector:
    def __init__(
        self,
        threshold: float = 0.5,
        hangover_frames: int = 8,   # ~256ms tail after speech ends (8 * 32ms)
    ) -> None:
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Voice detection is unavailable. Install the dependencies from requirements.txt."
            ) from exc

        self.model = load_silero_vad()
        self.threshold = threshold
        self.hangover_frames = hangover_frames
        self._silent_run = 0  # tracks consecutive non-speech frames

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        if audio_chunk.size == 0:
            return False

        import torch

        probabilities = []
        for offset in range(0, len(audio_chunk), 512):
            frame = audio_chunk[offset : offset + 512]
            if len(frame) < 512:
                break
            tensor = torch.from_numpy(frame.astype(np.float32, copy=False))
            probabilities.append(float(self.model(tensor, sample_rate).item()))

        raw_speech = bool(probabilities) and max(probabilities) >= self.threshold

        if raw_speech:
            self._silent_run = 0
            return True

        # Hangover: keep classifying as speech briefly after voice actually
        # stops, so trailing phonemes/consonants aren't chopped off before
        # the silence timer starts counting.
        if self._silent_run < self.hangover_frames:
            self._silent_run += 1
            return True

        return False