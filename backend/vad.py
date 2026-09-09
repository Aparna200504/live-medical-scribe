from __future__ import annotations

import numpy as np


class VoiceActivityDetector:
    def __init__(self, threshold: float = 0.5) -> None:
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Voice detection is unavailable. Install the dependencies from requirements.txt."
            ) from exc

        self.model = load_silero_vad()
        self.threshold = threshold

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
        return bool(probabilities) and max(probabilities) >= self.threshold