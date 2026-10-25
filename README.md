# SpeechGrade

SpeechGrade is a local speech evaluation prototype for timestamped feedback and reproducible progress tracking.

## Current scope

The scaffold provides the SQLAlchemy schema, SQLite initialization, a health endpoint, and an audio upload endpoint. Uploads are transcribed locally with faster-whisper (`base`, CPU/int8), then word-level filler and pause features are stored in SQLite. LLM scoring and the Streamlit dashboard are not implemented yet.

## Setup

Use Python 3.11, then from this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive docs are at `/docs`. Upload audio through `POST /speeches` as multipart form data. Accepted `speech_type` values are `ideal`, `flawed`, and `user_submission`. The endpoint stores the file under `data/`, saves the transcript and extracted features, and retains the upload if transcription fails. The first request downloads the configured Whisper model if it is not already cached.

The SQLite database is created at `speechgrade.db` on startup. To initialize it directly:

```powershell
python -c "from app.db import init_db; init_db()"
```
