# ClaimCheck Video AI

**AI-Powered Video Claim Extraction and Human Fact-Checking Portal**

## Overview

ClaimCheck Video AI is a research-oriented platform designed to process social media videos, extract potentially verifiable claims from spoken content, and support human reviewers in assessing those claims.

The system is designed to support Urdu and English video content.

## Main Features

- Video URL-based processing.
- Video transcription using Google Gemini.
- AI-powered claim extraction.
- Urdu and English content processing.
- Human claim review interface.
- Evidence and source recording.
- PostgreSQL database integration.
- Review submission and storage.
- Backend health monitoring.

## Technology Stack

### Frontend
- React.js
- Vite
- JavaScript
- CSS

### Backend
- Python
- FastAPI
- SQLAlchemy
- Pydantic
- Google Gemini API

### Database
- PostgreSQL

## Project Structure

```text
ClaimCheck Video/
├── backend/
│   ├── app/
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
├── README.md
└── .gitignore
```

*Adjust this structure to match the actual project folders.*

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ClaimCheck-Video-AI.git
cd ClaimCheck-Video-AI
```

### 2. Backend setup

```bash
cd backend
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the required environment variables in a local `.env` file. Do not upload this file to GitHub.

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

### 4. Database setup

Create a PostgreSQL database and configure its connection string in the backend environment variables.

Apply the project's database setup scripts or migrations, if provided.

## Running the Application

Start the FastAPI backend using the project's actual application entry point.

For example:

```bash
uvicorn app.main:app --reload
```

Start the frontend separately:

```bash
npm run dev
```

The backend API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Research Workflow

1. Submit a supported social media video URL.
2. Process the video and generate its transcript.
3. Extract potentially verifiable claims.
4. Review extracted claims through the human review interface.
5. Record review decisions and evidence.
6. Save review records in PostgreSQL.

## Security

- Keep API keys and database credentials in environment variables.
- Never commit `.env` files.
- Do not upload private research data or confidential videos.
- Use `.env.example` to document required configuration variables without real secrets.

## Project Status

The current version focuses on video processing, claim extraction, human review and database persistence.

Future development will include research dataset preparation, multilingual classification models, retrieval-augmented generation and model evaluation.

## License

Research project. All rights reserved unless a separate license is added.
