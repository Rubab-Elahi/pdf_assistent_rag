"""
main.py - FastAPI entry point for the RAG Bot (Vercel deployment)
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import load_qa_chain, ask

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
        # Load chain fresh per request — safe for Vercel serverless cold starts
        chain = load_qa_chain()
        answer = ask(req.question, chain)
        return {"answer": answer}
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="FAISS index not found. Ensure 'faiss_index/' is included in the deployment."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
