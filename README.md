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

## Architecture

```text
Browser Microphone
       ↓
Web Audio API
       ↓ PCM16
FastAPI WebSocket
       ↓
Silero VAD
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

### Speech Detection

`vad.py` uses **Silero VAD** to determine whether incoming PCM audio contains speech. A short hangover window prevents words from being cut off at the end of speech.

### Multilingual ASR

`asr.py` uses **Faster-Whisper (`base`)** with:

* CPU + INT8 inference
* Automatic language detection
* `condition_on_previous_text=False` to reduce language carry-over between segments
* Minimum segment duration filtering
* Language confidence handling

Each transcript segment contains:

```json
{
  "text": "Patient has fever since yesterday.",
  "language": "en",
  "language_name": "English"
}
```

Language detection is performed independently for each speech segment, allowing the session to switch languages multiple times.

### WebSocket Pipeline

`main.py` manages the continuous session:

1. Receive PCM16 audio
2. Run VAD
3. Buffer speech
4. Transcribe after silence or maximum segment duration
5. Send transcript and detected language to the frontend
6. Generate the final clinical summary when the session ends

The backend also exposes `/health`, `/health/ready`, `/status`, and `/metrics`.

### Clinical Summarization

`llm.py` sends the complete multilingual transcript to **Gemini**.

The prompt instructs the model to:

* Understand mixed-language conversations
* Extract only explicitly stated information
* Avoid inventing diagnoses or treatment
* Return structured English JSON

The response is validated using the `ClinicalSummary` Pydantic schema.

## Tech Stack

| Component  | Technology            |
| ---------- | --------------------- |
| Frontend   | HTML, CSS, JavaScript |
| Backend    | FastAPI, WebSockets   |
| VAD        | Silero VAD            |
| ASR        | Faster-Whisper        |
| LLM        | Gemini                |
| Validation | Pydantic              |
| Audio      | Web Audio API, PCM16  |
| Testing    | Pytest                |

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

Run:

```bash
uvicorn backend.main:app --reload
```

### Frontend

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

## Project Structure

```text
backend/
├── main.py       # WebSocket server and session management
├── asr.py        # Faster-Whisper transcription
├── vad.py        # Silero speech detection
├── llm.py        # Gemini summarization + Pydantic schemas
├── benchmark_threads.py # compares Faster-Whisper CPU inference time across different CPU thread counts to help tune ASR performance.
└── test_*.py     # Tests


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
* Clinical output is intended for documentation assistance and requires human review.

## Future Improvements

* GPU inference and model benchmarking
* Better language detection for short utterances
* Speaker diarization
* Persistent consultation storage
* Production authentication and deployment
