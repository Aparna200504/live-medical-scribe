# benchmark_threads.py
import time
import numpy as np
from faster_whisper import WhisperModel

# ~6 seconds of silence-ish audio; swap in a real sample for realistic numbers
audio = np.zeros(16000 * 6, dtype=np.float32)

for threads in [4, 6, 8, 10, 12]:
    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1)
    start = time.perf_counter()
    segments, info = model.transcribe(audio, beam_size=2, language=None, temperature=0)
    list(segments)  # force generator to run
    elapsed = time.perf_counter() - start
    print(f"cpu_threads={threads:>2}  ->  {elapsed:.2f}s")