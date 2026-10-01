# RAG PDF Assistant

A Retrieval-Augmented Generation (RAG) assistant for PDF document parsing, indexing, and intelligent Q&A powered by OpenAI, FAISS, Streamlit, and FastAPI.

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application (API entry point & Vercel deployment)
│   ├── rag.py               # Vector retrieval & OpenAI QA chain logic
│   ├── ingest.py            # PDF document parser & FAISS index builder
│   ├── faiss_index/         # Serialized FAISS vector store
│   ├── vercel.json          # Vercel deployment configuration
│   ├── .vercelignore        # Vercel deployment ignore rules
│   ├── requirements.txt     # Python package requirements for backend
│   └── *.pdf                # Document repository for ingestion
├── frontend/
│   ├── app.py               # Streamlit web application
│   └── requirements.txt     # Python package requirements for frontend
├── .env                     # API key configuration (OPENAI_API_KEY)
└── README.md                # Project documentation
```

## Setup & Usage

### 1. Environment Configuration
Create a `.env` file in the project root directory (or in `backend/`):
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Running the Backend (FastAPI API Server)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Running the Frontend (Streamlit App)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### 4. Running PDF Ingestion via CLI
```bash
cd backend
python ingest.py path/to/document.pdf
```
