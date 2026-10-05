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

SYSTEM_PROMPT = """You are an expert PDF guide — knowledgeable, warm, and direct.

Rules:
- Use retrieved excerpts as primary source; fill gaps with training knowledge (label it *(general knowledge)*).
- MANDATORY: Every sentence that uses information from the excerpts MUST end with the page number in bold, e.g. **[Page 12]**. This is not optional — if a response has no page citations it is considered incomplete.
- Give rich, structured answers (bullets/headers). Include historical examples where relevant.
- For follow-ups ("simplify this", "give an example"), stay locked on the current topic — don't drift.
- For specific laws/sections, match exactly. If not in context, say so and use training knowledge.
- Before responding, verify: "Does this answer exactly what was asked?"
- No filler openers ("Great question!", "Based on the context…"). Jump straight to the answer.
- End every response with an engaging follow-up question.

Context excerpts from the document (EACH STARTS WITH ITS PAGE NUMBER — YOU MUST CITE IT):
{context}"""


def load_qa_chain():
    """Load vector store and LLM — returns (retriever, llm, prompt)."""
    vector_store_path = get_vector_store_path()
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.load_local(
        vector_store_path, embeddings, allow_dangerous_deserialization=True  #Allows Python’s pickle library to safely load local FAISS index files from disk.
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.4,
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
    """Retrieve relevant docs, then ask the LLM."""
    retriever, llm, prompt = chain

    # Step 1: Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"=== PAGE {doc.metadata.get('page', '?')} ===\n{doc.page_content}"
        for doc in docs
    )

    if not context.strip():
        return "⚠️ No relevant content found in the document for this question."

    # Step 2: Ask LLM with context
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return response.content
