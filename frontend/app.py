import os
import glob
import shutil
import sys
import streamlit as st

# Add project root and backend directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from backend.ingest import ingest_pdf
    from backend.rag import load_qa_chain, ask
except ImportError:
    from ingest import ingest_pdf
    from rag import load_qa_chain, ask


# Page configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide",
)


def list_pdfs():
    """Find all PDF files in backend and frontend directories."""
    pdf_paths = glob.glob(os.path.join(BACKEND_DIR, "*.pdf")) + glob.glob(
        os.path.join(os.path.dirname(__file__), "*.pdf")
    )
    pdf_names = sorted(list(set([os.path.basename(p) for p in pdf_paths])))
    return pdf_names


def get_or_load_chain():
    """Load or retrieve the active QA chain stored in session state."""
    if "active_chain" not in st.session_state or st.session_state.active_chain is None:
        try:
            st.session_state.active_chain = load_qa_chain()
        except Exception:
            st.session_state.active_chain = None
    return st.session_state.active_chain


# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = (
        "💡 Select or upload a PDF, then click **Ingest & Process PDF**."
    )


# --- Custom CSS ---
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

# Header Section
st.markdown(
    """
    <div class="header-box">
        <h1>📚 AI RAG PDF Assistant</h1>
        <p>Upload or select a PDF document, ingest it, and ask questions powered by OpenAI & FAISS</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Two-Column Layout
left_col, right_col = st.columns([1, 2], gap="large")

# --- Left Sidebar / Column: PDF Management & Ingestion ---
with left_col:
    st.subheader("📄 Document Selection")

    pdfs = list_pdfs()
    selected_pdf = st.selectbox(
        "Select Existing PDF",
        options=["None"] + pdfs if pdfs else ["No PDFs found"],
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Or Upload New PDF",
        type=["pdf"],
    )

    if st.button("⚡ Ingest & Process PDF", use_container_width=True, type="primary"):
        target_path = None

        # Priority 1: Uploaded file
        if uploaded_file is not None:
            dest_path = os.path.join(BACKEND_DIR, uploaded_file.name)
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            target_path = dest_path

        # Priority 2: Selected existing PDF
        elif selected_pdf and selected_pdf not in ["None", "No PDFs found"]:
            target_path = os.path.join(BACKEND_DIR, selected_pdf)
            if not os.path.exists(target_path):
                target_path = os.path.join(os.path.dirname(__file__), selected_pdf)

        if not target_path or not os.path.exists(target_path):
            st.session_state.ingest_status = (
                "⚠️ **Error:** Please select an existing PDF or upload a new PDF file first."
            )
        else:
            pdf_name = os.path.basename(target_path)
            with st.spinner(f"Ingesting `{pdf_name}`..."):
                try:
                    ingest_pdf(target_path)
                    st.session_state.active_chain = load_qa_chain()
                    st.session_state.ingest_status = (
                        f"✅ **Successfully ingested `{pdf_name}`!**\n\n"
                        f"Vector index updated and ready for Q&A."
                    )
                    st.rerun()
                except Exception as e:
                    st.session_state.ingest_status = f"❌ **Ingestion failed:** {str(e)}"

    st.info(st.session_state.ingest_status)


# --- Right Column: Chat Interface ---
with right_col:
    st.subheader("💬 Document Q&A Chat")

    # Chat Header Controls
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with ctrl_col2:
        if st.button("🔄 Refresh PDF List", use_container_width=True):
            st.rerun()

    # Chat message container
    chat_container = st.container(height=450)

    # Render previous conversation history
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input box at the bottom of chat
    if user_message := st.chat_input("Type your question about the PDF document..."):
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": user_message})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_message)

        # Get active QA chain
        chain = get_or_load_chain()

        # Generate bot response
        if chain is None:
            bot_message = (
                "⚠️ **No document index found.** Please upload or select a PDF and click **Ingest & Process PDF** first."
            )
        else:
            try:
                with st.spinner("Thinking..."):
                    bot_message = ask(user_message, chain)
            except Exception as e:
                bot_message = f"❌ **Error generating response:** {str(e)}"

        # Append and display assistant response
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(bot_message)