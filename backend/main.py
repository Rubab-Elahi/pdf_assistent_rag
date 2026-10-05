"""
main.py - FastAPI entry point for the RAG Bot (Vercel deployment)
"""

import os
import shutil
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import load_qa_chain, ask, get_cached_chain, clear_chain_cache
from ingest import ingest_pdf

app = FastAPI(title="RAG PDF Assistant API")

# Enable CORS for external frontends (e.g., Streamlit, React, or local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"message": "RAG PDF Assistant API is live!"}


@app.post("/ask")
def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chain = get_cached_chain()
        answer = ask(req.question, chain)
        return {"answer": answer}
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="FAISS index not found. Ensure 'faiss_index/' is included in the deployment."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Accept a PDF upload, ingest it, and update the FAISS index."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        ingest_pdf(tmp_path)
        os.unlink(tmp_path)  # Clean up temp file
        clear_chain_cache()  # Reset chain cache to reload newly ingested vector store
        return {"message": f"Successfully ingested '{file.filename}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """List PDF files available in the backend directory."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    pdfs = [
        f for f in os.listdir(backend_dir)
        if f.endswith(".pdf")
    ]
    return {"documents": sorted(pdfs)}
