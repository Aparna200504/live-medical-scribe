# Live Medical Scribe

## Project Overview
**Live Medical Scribe** is a real-time medical documentation prototype that converts a doctor-patient conversation into a structured clinical summary.

The application:

- Captures microphone audio in the browser
- Filters silence using **Silero VAD**
- Converts speech to text using **Faster-Whisper**
- Displays the transcript in real time
- Uses **Google Gemini** to generate a structured clinical summary
- Validates the output using **Pydantic**

## Key Features

- Real-time microphone recording
- Voice Activity Detection (VAD)
- Speech-to-text transcription
- Live transcript display
- AI-powered clinical summary generation
- Structured medical information extraction
- Positive and negative symptom separation
- Pydantic schema validation
- Responsive web interface
- WebSocket-based communication
- Error handling and input validation

## Clinical Summary
The AI extracts:

- **Patient Details** — name, age, sex, identifiers
- **Chief Complaint**
- **History of Present Illness**
- **Symptoms** — positive and stated negatives
- **Past Medical History**
- **Medication History**
- **Clinical Observations**
- **Assessment**
- **Plan**
The system only extracts information present in the transcript and does not intentionally invent medical information. The generated summary should always be reviewed by a qualified clinician.

## Architecture

```
Microphone
	↓
Browser Audio Capture
	↓
FastAPI WebSocket
	↓
Silero VAD
	↓
Faster-Whisper
	↓
Live Transcript
	↓
Google Gemini
	↓
Pydantic Validation
	↓
Structured Clinical Summary
```

## Technology Stack
**Backend**

- Python
- FastAPI
- Uvicorn
- Faster-Whisper
- Silero VAD
- PyTorch
- Google Gemini API
- Pydantic
**Frontend**

- HTML5
- CSS3
- JavaScript
- Web Audio API
- WebSockets

## Project Structure

```
doctorsapp/
│
├── backend/
│   ├── main.py
│   ├── vad.py
│   ├── asr.py
│   ├── llm.py
│   ├── test_schema.py
│   ├── test_client.py
│   └── .env.example
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Virtual Environment
From the project root:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```
python -m pip install -r requirements.txt
```

### 3. Configure Gemini
Create:

```
backend/.env
```
Add:

```
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
Keep the API key private and do not commit `.env` to GitHub.

### 4. Start Backend

```
python -m uvicorn main:app --reload --app-dir backend
```
Backend:

```
http://localhost:8000
```

### 5. Start Frontend
Open another terminal:

```
cd frontend
python -m http.server 3000
```
Open:

```
http://localhost:3000
```
Allow microphone access when prompted.

## Usage

1. Open the application.
2. Click **Start Recording**.
3. Allow microphone access.
4. Speak during the consultation.
5. View the transcript as it is generated.
6. Click **Stop & Summarize**.
7. Review the generated clinical summary.

## Testing
Run the schema tests:

```
python -m pytest -q backend/test_schema.py
```
The tests verify valid and invalid clinical summary structures.


## Future Work

- Replace `ScriptProcessorNode` with `AudioWorklet`
- Add speaker diarization
- Support multiple languages
- Improve medical terminology recognition
- Add authentication and secure data handling
- Add automated end-to-end and performance testing
- Integrate with EHR systems
- Add production monitoring and scalable background workers

## Recruiter Summary
This project demonstrates practical experience in **AI/ML, Python backend development, real-time audio processing, WebSockets, speech recognition, LLM integration, structured data validation, and frontend development**. It showcases an end-to-end AI application from microphone input to a structured, user-facing medical document.
"# live-medical-scribe" 
