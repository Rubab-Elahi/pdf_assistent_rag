import os
import time
import uuid
import requests
import streamlit as st

# Backend API URL 
BACKEND_URL = os.getenv("BACKEND_URL", "https://pdf-assistent-rag.vercel.app")

# Page Configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = (
        "💡 Select or upload a PDF, then click **Ingest & Process PDF**."
    )

if "pdf_list" not in st.session_state:
    st.session_state.pdf_list = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0


# --- Helper API Functions ---
def stream_rag_backend(question, model="gpt-4o", temperature=0.0):
    """Stream response from backend /ask/stream endpoint, falling back to /ask if 404."""
    session_id = st.session_state.get("session_id", "default")
    try:
        res = requests.post(
            f"{BACKEND_URL}/ask/stream",
            json={"question": question, "model": model, "temperature": temperature, "session_id": session_id},
            stream=True,
            timeout=60,
        )
        if res.status_code == 200:
            for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
            return

        # Fallback to /ask if /ask/stream is not deployed yet on Vercel
        if res.status_code == 404:
            fallback_res = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question, "model": model, "temperature": temperature, "session_id": session_id},
                timeout=30,
            )
            if fallback_res.status_code == 200:
                answer = fallback_res.json().get("answer", "⚠️ No answer returned from API.")
                words = answer.split(" ")
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
                    time.sleep(0.015)
                return

        try:
            detail = res.json().get("detail", res.text)
        except Exception:
            detail = res.text
        yield f"❌ **Backend Error (HTTP {res.status_code}):** {detail}"
    except requests.exceptions.Timeout:
        yield "⏱️ **Request timed out.** Please try again."
    except requests.exceptions.ConnectionError:
        yield f"🔌 **Could not reach backend:** `{BACKEND_URL}`"
    except Exception as e:
        yield f"❌ **Error:** {str(e)}"


def query_rag_backend(question, model="gpt-4o", temperature=0.0):
    """Query the backend /ask endpoint."""
    session_id = st.session_state.get("session_id", "default")
    try:
        res = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": question, "model": model, "temperature": temperature, "session_id": session_id},
            timeout=30,
        )
        if res.status_code == 200:
            return res.json().get("answer", "⚠️ No answer returned from API.")
        return f"❌ **Backend Error (HTTP {res.status_code}):** {res.text}"
    except requests.exceptions.Timeout:
        return "⏱️ **Request timed out.** Please try again."
    except requests.exceptions.ConnectionError:
        return f"🔌 **Could not reach backend:** `{BACKEND_URL}`"
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


def fetch_pdf_list():
    """Fetch existing PDFs from backend /documents endpoint."""
    try:
        res = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if res.status_code == 200:
            return res.json().get("documents", [])
    except Exception:
        pass
    return []


def send_pdf_for_ingestion(uploaded_file):
    """Upload a PDF file to backend /ingest endpoint."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        res = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=60)
        if res.status_code == 200:
            return True, f"✅ **Successfully ingested `{uploaded_file.name}`!**\n\nVector index updated. You can now ask questions."
        else:
            try:
                detail = res.json().get("detail", res.text)
            except Exception:
                detail = res.text
            return False, f"❌ **Ingestion failed:** {detail}"
    except requests.exceptions.ConnectionError:
        return False, f"🔌 **Could not reach backend:** `{BACKEND_URL}`"
    except Exception as e:
        return False, f"❌ **Error:** {str(e)}"


# --- Clean Subtle Custom CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean container padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }

    /* Subtle App Header Bar */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .app-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-logo {
        font-size: 1.5rem;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 4px 8px;
        line-height: 1;
    }
    .app-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .app-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        margin: 0;
    }
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .online-indicator {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        display: inline-block;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .sidebar-section-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Empty state card */
    .empty-state {
        text-align: center;
        padding: 35px 20px;
        background: #FFFFFF;
        border: 1px dashed #CBD5E1;
        border-radius: 12px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .empty-state-icon {
        font-size: 2.2rem;
        margin-bottom: 8px;
    }
    .empty-state-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .empty-state-subtitle {
        font-size: 0.86rem;
        color: #64748B;
        max-width: 480px;
        margin: 0 auto;
    }

    /* Micro timing badge */
    .timing-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.74rem;
        color: #64748B;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 2px 7px;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Live timer shown during generation in bottom-right corner */
    .live-timer {
        position: fixed;
        bottom: 78px;
        right: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #4F46E5;
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        border-radius: 9999px;
        padding: 5px 14px;
        z-index: 9999;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.12);
        animation: pulse-timer 1s ease-in-out infinite alternate;
    }
    @keyframes pulse-timer {
        from { opacity: 0.85; }
        to   { opacity: 1; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: Document Management & Model Settings ---
with st.sidebar:
    st.markdown("<div class='sidebar-section-title'>⚙️ Model Configuration</div>", unsafe_allow_html=True)
    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    selected_model = st.selectbox(
        "OpenAI Model",
        options=model_options,
        index=0,
        help="Select which OpenAI LLM will generate responses.",
        label_visibility="collapsed",
    )

    selected_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Controls response creativity. 0.0 = strict & deterministic (best for RAG). 1.0 = more creative & varied.",
    )
    st.session_state.temperature = selected_temperature

    # Temperature label hint
    if selected_temperature == 0.0:
        st.caption("🎯 **Strict mode** — deterministic, best for RAG")
    elif selected_temperature <= 0.3:
        st.caption("✅ **Low** — mostly consistent answers")
    elif selected_temperature <= 0.6:
        st.caption("⚖️ **Balanced** — some variation in phrasing")
    else:
        st.caption("🎨 **Creative** — more varied, may drift from PDF")

    st.markdown("<div class='sidebar-section-title'>📄 Document Management</div>", unsafe_allow_html=True)

    if not st.session_state.pdf_list:
        st.session_state.pdf_list = fetch_pdf_list()

    pdf_list = st.session_state.pdf_list
    selected_pdf = st.selectbox(
        "Existing Documents",
        options=["None"] + pdf_list if pdf_list else ["No PDFs available"],
        index=0,
        help="Select a PDF already processed on the server.",
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a new PDF to extract and index into FAISS.",
    )

    if st.button("⚡ Ingest & Process PDF", use_container_width=True, type="primary"):
        if uploaded_file is not None:
            with st.spinner(f"Ingesting `{uploaded_file.name}`..."):
                success, message = send_pdf_for_ingestion(uploaded_file)
                st.session_state.ingest_status = message
                if success:
                    st.session_state.pdf_list = fetch_pdf_list()
        else:
            st.session_state.ingest_status = "⚠️ Please select or upload a PDF first."

    st.info(st.session_state.ingest_status)

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Refresh", use_container_width=True, help="Refresh document list"):
            st.session_state.pdf_list = fetch_pdf_list()
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
            # Clear in-memory messages
            st.session_state.messages = []
            # Clear PostgreSQL history for this session
            try:
                requests.delete(
                    f"{BACKEND_URL}/session/{st.session_state.session_id}",
                    timeout=10,
                )
            except Exception:
                pass
            # Generate a fresh session_id for the next conversation
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    st.caption(f"**Backend:** `{BACKEND_URL}`")


# --- Main Workspace: Clean Header & Chat ---
st.markdown(
    f"""
    <div class="app-header">
        <div class="app-brand">
            <span class="app-logo">📚</span>
            <div>
                <h1 class="app-title">RAG PDF Assistant</h1>
                <p class="app-subtitle">Strictly grounded document Q&A powered by FAISS & OpenAI</p>
            </div>
        </div>
        <div class="model-badge">
            <span class="online-indicator"></span>
            <span>{selected_model} &nbsp;|&nbsp; 🌡️ {selected_temperature}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chat Messages Container
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">💬</div>
                <div class="empty-state-title">Ask anything about your document</div>
                <div class="empty-state-subtitle">
                    Select or upload a PDF from the sidebar, then ask questions. Answers are strictly grounded in the document context with exact page citations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Starter Prompts
        st.markdown("<div style='font-size:0.82rem; font-weight:600; color:#64748B; margin-bottom:6px;'>SUGGESTED QUESTIONS:</div>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("📄 Summarize PDF", use_container_width=True):
                st.session_state.pending_query = "Please provide a comprehensive summary of this document."
                st.rerun()
        with col_s2:
            if st.button("💡 Key Takeaways", use_container_width=True):
                st.session_state.pending_query = "What are the main takeaways and key highlights in this document?"
                st.rerun()
        with col_s3:
            if st.button("📌 Main Topics", use_container_width=True):
                st.session_state.pending_query = "What are the core topics and sections covered in this document?"
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("role") == "assistant" and "elapsed_time" in msg:
                temp_val = msg.get("temperature", 0.0)
                st.markdown(
                    f"<div class='timing-badge'>⚡ {msg.get('model', selected_model)} &nbsp;|&nbsp; 🌡️ {temp_val} &nbsp;|&nbsp; ⏱️ {msg['elapsed_time']:.2f}s</div>",
                    unsafe_allow_html=True,
                )

# Live timer placeholder — shown in bottom-right corner while model generates
live_timer = st.empty()

# Determine query from chat_input or a clicked suggestion
active_query = st.session_state.pending_query
if active_query:
    st.session_state.pending_query = None
else:
    active_query = st.chat_input("Type your question about the PDF document...")

def timed_stream(gen, timer_placeholder, start):
    """Wrap a generator to update a live timer placeholder on every token."""
    for chunk in gen:
        elapsed = time.perf_counter() - start
        timer_placeholder.markdown(
            f"<div class='live-timer'>⏱️ {elapsed:.1f}s &nbsp; <span style='color:#94A3B8;font-weight:400;'>generating...</span></div>",
            unsafe_allow_html=True,
        )
        yield chunk

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            start_time = time.perf_counter()
            bot_message = st.write_stream(
                timed_stream(stream_rag_backend(active_query, model=selected_model, temperature=selected_temperature), live_timer, start_time)
            )
            elapsed_time = time.perf_counter() - start_time
            # Clear the live timer once complete
            live_timer.empty()
            st.markdown(
                f"<div class='timing-badge'>⚡ {selected_model} &nbsp;|&nbsp; 🌡️ {selected_temperature} &nbsp;|&nbsp; ⏱️ {elapsed_time:.2f}s</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_message,
        "model": selected_model,
        "temperature": selected_temperature,
        "elapsed_time": elapsed_time,
    })