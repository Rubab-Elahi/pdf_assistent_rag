import os
import requests
import streamlit as st

# Backend API URL 
BACKEND_URL = os.getenv("BACKEND_URL", "https://pdf-assistent-rag.vercel.app")

# Page Configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide",
)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = (
        "💡 Select or upload a PDF, then click **Ingest & Process PDF**."
    )


# --- Helper API Functions ---
def query_rag_backend(question):
    """Query the backend /ask endpoint."""
    try:
        res = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": question},
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


# --- Custom CSS & Header ---
st.markdown(
    """
    <style>
    .header-box { text-align: center; margin-bottom: 25px; }
    .header-box h1 { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 6px; }
    .header-box p { font-size: 1.05rem; color: #64748B; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header-box">
        <h1>📚 AI RAG PDF Assistant</h1>
        <p>Upload or select a PDF document, ingest it, and ask questions powered by OpenAI & FAISS</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Layout ---
left_col, right_col = st.columns([1, 2], gap="large")

# Left Column: Document Management
with left_col:
    st.subheader("📄 Document Selection")

    if "pdf_list" not in st.session_state:
        st.session_state.pdf_list = []

    pdf_list = st.session_state.pdf_list
    selected_pdf = st.selectbox(
        "Select Existing PDF",
        options=["None"] + pdf_list if pdf_list else ["No PDFs found"],
        index=0,
    )

    uploaded_file = st.file_uploader("Or Upload New PDF", type=["pdf"])

    if st.button("⚡ Ingest & Process PDF", use_container_width=True, type="primary"):
        if uploaded_file is not None:
            with st.spinner(f"Ingesting `{uploaded_file.name}` on backend..."):
                success, message = send_pdf_for_ingestion(uploaded_file)
                st.session_state.ingest_status = message
        else:
            st.session_state.ingest_status = "⚠️ Please upload a PDF file first."

    st.info(st.session_state.ingest_status)
    st.caption(f"**Backend:** `{BACKEND_URL}`")


# Right Column: Chat Interface
with right_col:
    st.subheader("💬 Document Q&A Chat")

    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with ctrl_col2:
        if st.button("🔄 Refresh PDF List", use_container_width=True):
            st.session_state.pdf_list = fetch_pdf_list()
            st.rerun()

    chat_container = st.container(height=450)

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_message := st.chat_input("Type your question about the PDF document..."):
        st.session_state.messages.append({"role": "user", "content": user_message})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_message)

        with st.spinner("Thinking..."):
            bot_message = query_rag_backend(user_message)

        st.session_state.messages.append({"role": "assistant", "content": bot_message})
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(bot_message)