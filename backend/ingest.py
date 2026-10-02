"""
ingest.py - Load a PDF, split into chunks, embed with OpenAI, save to FAISS
"""

import os
import pymupdf as fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

if os.getenv("VERCEL"):
    VECTOR_STORE_PATH = "/tmp/faiss_index"
else:
    VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
EMBED_MODEL = "text-embedding-3-small"


def extract_text_from_pdf(pdf_path: str) -> list[Document]:
    """
    Manually extract text from every page of a PDF using PyMuPDF (fitz).
    Returns a list of LangChain Document objects (one per page).
    Skips pages that have no extractable text.
    """
    documents = []
    pdf = fitz.open(pdf_path)

    print(f"   Total pages in PDF: {len(pdf)}")
    skipped = 0

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("text")  # plain text extraction

        if not text.strip():
            skipped += 1
            continue  # skip blank / image-only pages

        doc = Document(
            page_content=text,
            metadata={
                "source": pdf_path,
                "page": page_num + 1,   # 1-indexed page number
            }
        )
        documents.append(doc)

    pdf.close()

    if skipped:
        print(f"   ⚠️  Skipped {skipped} blank/image-only pages.")

    print(f"   ✅ Extracted text from {len(documents)} pages.")
    return documents


def ingest_pdf(pdf_path: str):
    """Extract PDF text → chunk → embed → save FAISS index."""

    print(f"📄 Loading PDF: {pdf_path}")

    # Step 1: Manually extract raw text from every page
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        raise ValueError(
            "❌ No text could be extracted from this PDF.\n"
            "   The file may be scanned images only. "
            "Try an OCR tool like Adobe Acrobat or tesseract first."
        )

    # Step 2: Split into overlapping chunks using LangChain text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # max characters per chunk
        chunk_overlap=200,  # overlap between consecutive chunks
        separators=["\n\n", "\n", ".", " ", ""],  # split priority order
    )
    chunks = splitter.split_documents(pages)
    print(f"   📦 Created {len(chunks)} chunks from {len(pages)} pages.")

    if not chunks:
        raise ValueError("❌ Text was found but produced 0 chunks. Check chunk_size settings.")

    # Step 3: Embed with OpenAI
    print("🤖 Embedding chunks with OpenAI...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Step 4: Build and save FAISS vector store
    print("💾 Building FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ Done! Vector store saved to '{VECTOR_STORE_PATH}/'")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/your.pdf")
    else:
        ingest_pdf(sys.argv[1])
