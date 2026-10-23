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

SYSTEM_PROMPT = """You are a friendly, helpful, and strictly grounded RAG pdf Assistant. Your job is to assist users by answering their questions using the saved data of the currently ingested PDF in the database.

Please follow these guidelines carefully:
1. SEARCH THE SAVED DATA: Carefully examine all the Context Excerpts provided below from the currently ingested PDF in the database.
2. WHEN THE ANSWER IS PRESENT:
   - Provide a clear, polite, and user-friendly answer directly addressing the user's query.
   - Base your answer strictly on the facts stated in the excerpts—never use outside knowledge, assumptions, or fabricated information.
   - Always cite the source page number(s) in bold for the information you provide (e.g., **[Page 2]**, **[Page 5]**).
3. WHEN THE ANSWER IS NOT PRESENT:
   - If the requested information is not mentioned or cannot be found in the saved data, politely and clearly let the user know that the answer is not present in the currently ingested document (for example: "I reviewed the saved document data, but unfortunately, the answer to your question is not present in the currently ingested PDF. Please let me know if you would like help with any topics covered in the document!").
   - Do NOT attempt to guess, extrapolate, or bring in external knowledge.
4. CONVERSATION HISTORY: You are also provided with the recent conversation history (if any). You MAY use it to understand follow-up questions, recall what you previously said, or maintain continuity. However, do NOT use history to bring in facts beyond what the PDF context provides.
5. TONE & FORMATTING:
   - Maintain a warm, courteous, and professional tone.
   - Use clean formatting (bullet points, clear paragraphs) to make your response easy and pleasant to read.

Context Excerpts from the currently ingested PDF:
{context}

Recent Conversation History (current session):
{history}"""


# Module-level caches
_retriever_cache = None
_llms_cache = {}
_prompt_cache = None
_chain_cache = None


def get_retriever():
    """Load or return cached FAISS retriever."""
    global _retriever_cache
    if _retriever_cache is None:
        vector_store_path = get_vector_store_path()
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        vector_store = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True  # Allows Python's pickle library to safely load local FAISS index files from disk.
        )
        _retriever_cache = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
        )
    return _retriever_cache


def get_prompt():
    """Return prompt template (not cached — history varies per call)."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])


def get_llm(model: str = "gpt-4o", temperature: float = 0.0):
    """Return cached ChatOpenAI instance for the requested model + temperature combo."""
    cache_key = f"{model}__temp{temperature}"
    if cache_key not in _llms_cache:
        _llms_cache[cache_key] = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llms_cache[cache_key]


def load_qa_chain(model: str = "gpt-4o", temperature: float = 0.0):
    """Load vector store and LLM for specified model/temperature — returns (retriever, llm, prompt)."""
    retriever = get_retriever()
    llm = get_llm(model, temperature)
    prompt = get_prompt()
    return retriever, llm, prompt


def clear_chain_cache():
    """Clear cached retriever and LLMs to force reloading FAISS index on next query."""
    global _retriever_cache, _llms_cache, _chain_cache
    _retriever_cache = None
    _llms_cache.clear()
    _chain_cache = None


def get_cached_chain(model: str = "gpt-4o", temperature: float = 0.0):
    """Return the QA chain for the specified model/temperature, using cached components."""
    return load_qa_chain(model, temperature)


def _format_history(history: list[dict]) -> str:
    """Format a list of {role, content} dicts into a readable conversation string."""
    if not history:
        return "(No previous conversation in this session.)"
    lines = []
    for turn in history:
        role_label = "Human" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {turn['content']}")
    return "\n".join(lines)


def ask(question: str, chain, history: list[dict] = None) -> str:
    """Retrieve top-k relevant docs, then ask the LLM strictly using context + session history."""
    retriever, llm, prompt = chain
    history = history or []

    # Step 1: Retrieve top-k chunks
    docs = retriever.invoke(question)

    if not docs:
        return "I'm sorry, but the answer to your question is not present in the saved data of the currently ingested PDF."

    context = "\n\n".join(
        f"=== PAGE {doc.metadata.get('page', '?')} ===\n{doc.page_content}"
        for doc in docs
    )

    if not context.strip():
        return "I'm sorry, but the answer to your question is not present in the saved data of the currently ingested PDF."

    # Step 2: Ask LLM with context + history
    history_text = _format_history(history)
    messages = prompt.format_messages(context=context, history=history_text, question=question)
    response = llm.invoke(messages)
    return response.content


def ask_stream(question: str, chain, history: list[dict] = None):
    """Retrieve top-k relevant docs, then stream LLM response tokens using context + session history."""
    retriever, llm, prompt = chain
    history = history or []

    # Step 1: Retrieve top-k chunks
    docs = retriever.invoke(question)

    if not docs:
        yield "I'm sorry, but the answer to your question is not present in the saved data of the currently ingested PDF."
        return

    context = "\n\n".join(
        f"=== PAGE {doc.metadata.get('page', '?')} ===\n{doc.page_content}"
        for doc in docs
    )

    if not context.strip():
        yield "I'm sorry, but the answer to your question is not present in the saved data of the currently ingested PDF."
        return

    # Step 2: Stream tokens from LLM with context + history
    history_text = _format_history(history)
    messages = prompt.format_messages(context=context, history=history_text, question=question)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
