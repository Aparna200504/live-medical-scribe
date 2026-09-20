# Live Medical Scribe

A multilingual, real-time medical scribe that captures a doctor-patient conversation, detects language changes, transcribes speech, and generates a structured clinical summary.

## Features

* 🎙️ Live microphone transcription over WebSocket
* 🌐 Multilingual speech recognition with automatic language detection
* 🔄 Supports switching between English, Hindi, Marathi and other supported languages within one session
* 🧠 Gemini-based structured clinical summarization
* 🛡️ Pydantic validation for generated summaries
* 🔇 Silero VAD to reduce unnecessary ASR processing
* 📊 Health, status and metrics endpoints
* ⚡ CPU-optimized Faster-Whisper inference
* 🧩 Stream-based speech buffering with silence detection and pre-roll audio

## Architecture

```text
Browser Microphone
       ↓
Web Audio API
       ↓ PCM16
FastAPI WebSocket
       ↓
StreamProcessor
       ↓
Silero VAD
       ↓
Speech Buffer
       ↓
Faster-Whisper
       ↓
Language Detection
       ↓
Transcript Segments
       ↓
Gemini
       ↓
Pydantic ClinicalSummary
       ↓
Frontend
```

## Implementation

### Speech Detection and Stream Processing

`stream_processor.py` manages the real-time audio processing pipeline.

It is responsible for:

* Running Silero VAD on incoming audio chunks
* Detecting speech and silence
* Buffering speech into phrases
* Ending a phrase after a configurable silence duration
* Limiting the maximum phrase duration
* Maintaining a short pre-roll buffer to help preserve the beginning of speech
* Resetting language state at phrase boundaries

This keeps audio-stream processing separate from the WebSocket session management handled by `main.py`.

### Voice Activity Detection

`vad.py` uses **Silero VAD** to determine whether incoming PCM audio contains speech.

A short hangover/silence window prevents speech from being cut off immediately when there is a brief pause between words.

VAD also prevents unnecessary ASR processing during periods of silence.

### Multilingual ASR

`asr.py` uses **Faster-Whisper (`base`)** with:

* CPU + INT8 inference
* Automatic language detection
* `condition_on_previous_text=False` to reduce language carry-over between segments
* Short-segment filtering
* Language confidence handling
* Configurable CPU thread count

Each completed speech phrase is transcribed independently.

Example:

```json
{
  "text": "Patient has fever since yesterday.",
  "language": "en",
  "language_name": "English"
}
```

Language detection is performed for each speech phrase, allowing a consultation to switch between languages multiple times within the same session.

For example:

```text
English → Hindi → Marathi → English
```

The language state is reset at phrase boundaries so that a previous phrase does not unnecessarily force the next phrase into the same language.

### WebSocket Pipeline

`main.py` manages the overall WebSocket session and delegates audio processing to `StreamProcessor`.

The session flow is:

1. Receive PCM16 audio from the browser
2. Pass the audio chunk to `StreamProcessor`
3. Run VAD and manage speech buffering
4. Detect the end of a speech phrase
5. Transcribe the completed phrase with Faster-Whisper
6. Send the transcript and detected language to the frontend
7. Collect transcript segments for the session
8. Generate the final clinical summary when the session ends

The backend also exposes:

* `/health`
* `/health/live`
* `/health/ready`
* `/status`
* `/metrics`

A heartbeat mechanism is used to monitor active WebSocket connections.

### Clinical Summarization

`llm.py` sends the complete multilingual transcript to **Gemini** after the consultation session ends.

The prompt instructs the model to:

* Understand mixed-language conversations
* Extract only explicitly stated information
* Avoid inventing diagnoses or treatment
* Return structured English JSON

The response is validated using the `ClinicalSummary` Pydantic schema before being returned to the frontend.

### CPU Performance Tuning

`benchmark_threads.py` compares Faster-Whisper CPU inference time across different CPU thread counts.

The benchmark was used to identify a practical thread configuration for local CPU inference.

The current configuration uses **8 CPU threads** for Faster-Whisper based on local benchmarking.

## Tech Stack

| Component         | Technology            |
| ----------------- | --------------------- |
| Frontend          | HTML, CSS, JavaScript |
| Backend           | FastAPI, WebSockets   |
| Stream Processing | Python, NumPy         |
| VAD               | Silero VAD            |
| ASR               | Faster-Whisper        |
| LLM               | Gemini                |
| Validation        | Pydantic              |
| Audio             | Web Audio API, PCM16  |
| Testing           | Pytest                |

## Setup

### Backend

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Run from the project root:

```bash
python -m uvicorn backend.main:app --reload
```

### Frontend

In a separate terminal:

```bash
cd frontend
python -m http.server 3000
```

Open:

```text
http://localhost:3000
```

## Testing

```bash
pytest
```

The project includes tests for the clinical summary schema and WebSocket client behavior.

The WebSocket integration test requires the FastAPI backend to be running.

## Project Structure

```text
backend/
├── main.py                  # FastAPI server and session management
├── asr.py                   # Faster-Whisper transcription and language detection
├── vad.py                   # Silero speech detection
├── stream_processor.py      # VAD, buffering, silence detection and phrase processing
├── llm.py                   # Gemini summarization + Pydantic schemas
├── benchmark_threads.py     # CPU thread performance benchmarking
└── test_*.py                # Tests

frontend/
├── index.html
├── app.js
└── styles.css

requirements.txt
README.md
```

## Limitations

* CPU inference is slower than GPU inference, especially with larger Whisper models.
* Language detection can become less reliable for very short or noisy speech segments.
* Brief language switches may be difficult to detect reliably when there is insufficient speech context.
* Clinical output is intended for documentation assistance and requires human review.

## Future Improvements

* GPU inference and model benchmarking
* Better language detection for short utterances
* Temporal smoothing for language-switch detection
* Speaker diarization
* Persistent consultation storage
* Production authentication and deployment
* More comprehensive multilingual ASR evaluation

---

## Backend Architecture

### `main.py`

Responsibilities:

* FastAPI application
* HTTP health/status/metrics endpoints
* WebSocket session lifecycle
* Audio reception and PCM conversion
* StreamProcessor integration
* Transcript collection
* Gemini clinical summary generation
* Heartbeat and connection monitoring

### `stream_processor.py`

Responsibilities:

* Silero VAD
* Speech buffering
* Silence-based phrase detection
* Maximum phrase duration
* Pre-roll audio
* Language reset at phrase boundaries

### `asr.py`

Responsibilities:

* Faster-Whisper inference
* Speech transcription
* Automatic language detection
* Language confidence handling

### `llm.py`

Responsibilities:

* Gemini clinical summarization
* Structured output handling
* Pydantic schema validation
* Clinical extraction guardrails

