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
2. Read the Context Excerpts carefully. If the excerpts DO NOT contain the answer to the user's question, respond strictly with: "The answer to this question is not available in the provided document."
3. You are FORBIDDEN from using any external knowledge, general world memory, training data, or outside assumptions.
4. MANDATORY CITATIONS: Every piece of information you state MUST include its page number in bold, e.g., **[Page 5]**.
5. Go straight to the answer without meta-introductions like "Based on the context...".
6. End your response with a concise, relevant follow-up question based on the document content.

Context Excerpts from the ingested document:
{context}"""


def load_qa_chain():
    """Load vector store and LLM — returns (retriever, llm, prompt)."""
    vector_store_path = get_vector_store_path()
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.load_local(
        vector_store_path, embeddings, allow_dangerous_deserialization=True  # Allows Python's pickle library to safely load local FAISS index files from disk.
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,  # Zero temperature for strict, deterministic grounding
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return retriever, llm, prompt

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
    """Retrieve top-k relevant docs, then ask the LLM strictly using context."""
    retriever, llm, prompt = chain

    # Step 1: Retrieve top-k chunks
    docs = retriever.invoke(question)

    if not docs:
        return "The answer to this question is not available in the provided document."

    context = "\n\n".join(
        f"=== PAGE {doc.metadata.get('page', '?')} ===\n{doc.page_content}"
        for doc in docs
    )

    if not context.strip():
        return "The answer to this question is not available in the provided document."

    # Step 2: Ask LLM strictly using context
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return response.content
