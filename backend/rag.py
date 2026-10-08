"""
rag.py - Load FAISS index and answer questions using OpenAI GPT
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_vector_store_path():
    """Return /tmp/faiss_index if it exists, otherwise return local bundled faiss_index."""
    tmp_path = "/tmp/faiss_index"
    bundled_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
    
    if os.path.exists(tmp_path):
        return tmp_path
    if os.path.exists(bundled_path):
        return bundled_path
    return tmp_path

EMBED_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """You are a strict, grounded PDF question-answering assistant.

RULES YOU MUST FOLLOW STRICTLY:
1. You MUST answer the user's question using ONLY the facts and details directly stated in the Context Excerpts below.
2. If the answer to the question cannot be found within the provided Context Excerpts, respond strictly with: "The answer to this question is not available in the provided document."
3. Do NOT use outside knowledge, general world memory, or assumptions under any circumstances.
4. MANDATORY CITATIONS: For every answer derived from the context, include the page citation in bold, e.g., **[Page 5]**.
5. Keep your response direct and clear without meta-introductions like "Based on the context...".
6. End your response with a concise, relevant follow-up question based on the document.

Context Excerpts from the ingested document:
{context}"""


# Similarity distance threshold — FAISS uses squared L2 distance; lower = more similar.
# Relevant chunks for OpenAI text-embedding-3-small typically fall between 0.70 and 1.25.
# Completely unrelated questions have L2 distance > 1.35.
SIMILARITY_THRESHOLD = 1.30


def load_qa_chain():
    """Load vector store and LLM — returns (vector_store, llm, prompt)."""
    vector_store_path = get_vector_store_path()
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.load_local(
        vector_store_path, embeddings, allow_dangerous_deserialization=True  # Allows Python's pickle library to safely load local FAISS index files from disk.
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,  # Zero temperature for strict, deterministic answers
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return vector_store, llm, prompt

# Cache chain at module level — avoids reloading FAISS on every request
_chain_cache = None


def clear_chain_cache():
    """Clear the cached QA chain to force reloading FAISS index on next query."""
    global _chain_cache
    _chain_cache = None


def get_cached_chain():
    """Return the cached QA chain, loading it once on first call."""
    global _chain_cache
    if _chain_cache is None:
        _chain_cache = load_qa_chain()
    return _chain_cache


def ask(question: str, chain) -> str:
    """Retrieve relevant docs with score filtering, then ask the LLM."""
    vector_store, llm, prompt = chain

    # Step 1: Retrieve top-k chunks with similarity scores (FAISS L2 distance)
    docs_with_scores = vector_store.similarity_search_with_score(question, k=8)

    # Step 2: Filter out chunks that are too dissimilar (high L2 distance = irrelevant)
    relevant_docs = [
        doc for doc, score in docs_with_scores
        if score <= SIMILARITY_THRESHOLD
    ]

    if not relevant_docs:
        return "The answer to this question is not available in the provided document."

    context = "\n\n".join(
        f"=== PAGE {doc.metadata.get('page', '?')} ===\n{doc.page_content}"
        for doc in relevant_docs
    )

    # Step 3: Ask LLM strictly using the filtered context
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return response.content
