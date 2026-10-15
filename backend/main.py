"""
main.py - FastAPI entry point for the RAG Bot (Vercel deployment)
"""

import os
import shutil
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag import load_qa_chain, ask, ask_stream, get_cached_chain, clear_chain_cache
from ingest import ingest_pdf
from database import init_db, save_message, get_history, clear_session

load_dotenv()

app = FastAPI(title="RAG PDF Assistant API")

# Enable CORS for external frontends (e.g., Streamlit, React, or local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Initialise the PostgreSQL database on app startup."""
    try:
        init_db()
    except Exception as e:
        print(f"⚠️  DB init warning: {e}")


class QueryRequest(BaseModel):
    question: str
    stream: bool = False
    model: str = "gpt-4o"
    session_id: str = "default"   # UUID sent by the frontend per browser session


@app.get("/")
def read_root():
    return {"message": "RAG PDF Assistant API is live!"}


@app.post("/ask")
def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chain = get_cached_chain(model=req.model)

        # Load recent history for this session
        history = get_history(req.session_id, limit=10)

        if req.stream:
            def generate():
                full_response = []
                for chunk in ask_stream(req.question, chain, history=history):
                    full_response.append(chunk)
                    yield chunk
                # Persist both turns after streaming completes
                save_message(req.session_id, "user", req.question)
                save_message(req.session_id, "assistant", "".join(full_response))

            return StreamingResponse(generate(), media_type="text/plain")

        answer = ask(req.question, chain, history=history)

        # Persist both turns
        save_message(req.session_id, "user", req.question)
        save_message(req.session_id, "assistant", answer)

        return {"answer": answer}
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="FAISS index not found. Ensure 'faiss_index/' is included in the deployment."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
def query_rag_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chain = get_cached_chain(model=req.model)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="FAISS index not found. Ensure 'faiss_index/' is included in the deployment."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Load recent history for this session
    history = get_history(req.session_id, limit=10)

    def generate():
        full_response = []
        try:
            for chunk in ask_stream(req.question, chain, history=history):
                full_response.append(chunk)
                yield chunk
        except Exception as e:
            yield f"\n\n[Error during generation: {str(e)}]"
        finally:
            # Always persist the turns (even on partial responses)
            if full_response:
                save_message(req.session_id, "user", req.question)
                save_message(req.session_id, "assistant", "".join(full_response))

    return StreamingResponse(generate(), media_type="text/plain")


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
        os.unlink(tmp_path)          # Clean up temp file
        clear_chain_cache()          # Reset chain cache to reload newly ingested vector store
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


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clear all chat history for a given session from PostgreSQL."""
    try:
        clear_session(session_id)
        return {"message": f"Session '{session_id}' history cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
