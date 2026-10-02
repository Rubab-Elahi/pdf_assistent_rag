"""
rag.py - Load FAISS index and answer questions using OpenAI GPT
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if os.getenv("VERCEL"):
    VECTOR_STORE_PATH = "/tmp/faiss_index"
else:
    VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
EMBED_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a document.
Use ONLY the context provided below to answer. Be detailed and informative.
If the context does not contain the answer, say so clearly.

Context from document:
{context}"""


def load_qa_chain():
    """Load vector store and LLM — returns (retriever, llm, prompt)."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True  #Allows Python’s pickle library to safely load local FAISS index files from disk.
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return retriever, llm, prompt


def ask(question: str, chain) -> str:
    """Retrieve relevant docs, then ask the LLM."""
    retriever, llm, prompt = chain

    # Step 1: Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    if not context.strip():
        return "⚠️ No relevant content found in the document for this question."

    # Step 2: Ask LLM with context
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return response.content
