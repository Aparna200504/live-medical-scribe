from __future__ import annotations

import numpy as np

from asr import Transcriber
from vad import VoiceActivityDetector


class StreamProcessor:
    """
    Converts continuous PCM audio chunks into phrase-level
    transcription events.

    Responsibilities:
    - VAD-based speech detection
    - speech buffering
    - silence-based endpointing
    - maximum phrase duration
    - short pre-roll to protect word boundaries
    - language re-detection for each phrase
    """

    def __init__(
        self,
        transcriber: Transcriber,
        vad: VoiceActivityDetector,
        sample_rate: int = 16000,
        max_speech_duration_s: float = 10.0,
        min_silence_duration_s: float = 0.6,
        pre_roll_duration_s: float = 0.2,
    ) -> None:

        self.transcriber = transcriber
        self.vad = vad
        self.sample_rate = sample_rate

        self.max_speech_samples = int(
            max_speech_duration_s * sample_rate
        )

        self.min_silence_samples = int(
            min_silence_duration_s * sample_rate
        )

        self.pre_roll_samples = int(
            pre_roll_duration_s * sample_rate
        )

        self._speech_buffer = np.array(
            [],
            dtype=np.float32,
        )

        self._pre_roll_buffer = np.array(
            [],
            dtype=np.float32,
        )

        self._silence_samples_count = 0
        self._is_speaking = False

    def process_chunk(
        self,
        chunk: np.ndarray,
    ) -> dict | None:

        if chunk.size == 0:
            return None

        if chunk.dtype != np.float32:
            chunk = chunk.astype(
                np.float32,
                copy=False,
            )

        is_speech = self.vad.is_speech(
            chunk,
            self.sample_rate,
        )

        if is_speech:

            # New speech phrase.
            if not self._is_speaking:

                self._is_speaking = True
                self._silence_samples_count = 0

                # Allow Whisper to detect the language
                # independently for this new phrase.
                self.transcriber.unlock_language()

                if len(self._pre_roll_buffer) > 0:
                    self._speech_buffer = np.concatenate(
                        (
                            self._pre_roll_buffer,
                            chunk,
                        )
                    )
                else:
                    self._speech_buffer = chunk.copy()

            else:

                self._speech_buffer = np.concatenate(
                    (
                        self._speech_buffer,
                        chunk,
                    )
                )

            self._silence_samples_count = 0

            if (
                len(self._speech_buffer)
                >= self.max_speech_samples
            ):
                return self._flush_and_transcribe(
                    reset_language=True
                )

            return None

        # -----------------------------
        # SILENCE
        # -----------------------------

        # Maintain a short pre-roll window even while
        # the speaker is silent.
        self._update_pre_roll(chunk)

        if not self._is_speaking:
            return None

        self._silence_samples_count += len(chunk)

        if (
            self._silence_samples_count
            >= self.min_silence_samples
        ):
            return self._flush_and_transcribe(
                reset_language=True
            )

        return None

    def _update_pre_roll(
        self,
        chunk: np.ndarray,
    ) -> None:

        if self.pre_roll_samples <= 0:
            return

        self._pre_roll_buffer = np.concatenate(
            (
                self._pre_roll_buffer,
                chunk,
            )
        )

        if len(self._pre_roll_buffer) > self.pre_roll_samples:
            self._pre_roll_buffer = (
                self._pre_roll_buffer[
                    -self.pre_roll_samples:
                ]
            )

    def _flush_and_transcribe(
        self,
        reset_language: bool = True,
    ) -> dict | None:

        if len(self._speech_buffer) == 0:
            return None

        audio = self._speech_buffer.copy()

        self._speech_buffer = np.array(
            [],
            dtype=np.float32,
        )

        self._silence_samples_count = 0
        self._is_speaking = False

        if reset_language:
            self.transcriber.unlock_language()

        result = self.transcriber.transcribe(
            audio,
            self.sample_rate,
        )

        if not result.get("text"):
            return None

        return result

    def finish(self) -> dict | None:

        if len(self._speech_buffer) == 0:
            return None

        return self._flush_and_transcribe(
            reset_language=True
        )