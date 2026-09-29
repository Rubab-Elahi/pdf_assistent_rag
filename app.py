"""
app.py - Gradio Web UI for RAG Bot
"""

import os
import glob
import shutil
import gradio as gr
from ingest import ingest_pdf
from rag import load_qa_chain, ask


def list_pdfs():
    """Find all PDF files in the current working directory."""
    pdf_paths = glob.glob(os.path.join(os.path.dirname(__file__), "*.pdf"))
    pdf_names = [os.path.basename(p) for p in pdf_paths]
    return pdf_names if pdf_names else ["No PDFs found"]


# Global variable to store active QA chain
active_chain = None


def get_or_load_chain():
    global active_chain
    if active_chain is None:
        try:
            active_chain = load_qa_chain()
        except Exception:
            active_chain = None
    return active_chain


def process_ingestion(selected_pdf, uploaded_file):
    global active_chain
    target_path = None

    # Priority 1: Newly uploaded file
    if uploaded_file is not None:
        uploaded_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
        filename = os.path.basename(uploaded_path)
        dest_path = os.path.join(os.path.dirname(__file__), filename)
        if uploaded_path != dest_path:
            shutil.copy(uploaded_path, dest_path)
        target_path = dest_path
    # Priority 2: Selected existing PDF
    elif selected_pdf and selected_pdf != "No PDFs found":
        target_path = os.path.join(os.path.dirname(__file__), selected_pdf)

    if not target_path or not os.path.exists(target_path):
        return (
            "⚠️ **Error:** Please select an existing PDF or upload a new PDF file first.",
            gr.update(choices=list_pdfs()),
        )

    try:
        pdf_name = os.path.basename(target_path)
        ingest_pdf(target_path)
        
        # Reload vector store and QA chain
        active_chain = load_qa_chain()
        
        updated_pdfs = list_pdfs()
        status_msg = (
            f"✅ **Successfully ingested `{pdf_name}`!**\n\n"
            f"Vector index updated and ready for Q&A."
        )
        return status_msg, gr.update(choices=updated_pdfs, value=pdf_name)
    except Exception as e:
        return f"❌ **Ingestion failed:** {str(e)}", gr.update(choices=list_pdfs())


def bot_respond(user_message, history):
    if not user_message.strip():
        return "", history

    chain = get_or_load_chain()
    if chain is None:
        bot_message = "⚠️ **No document index found.** Please upload or select a PDF and click **Ingest & Process PDF** first."
    else:
        try:
            bot_message = ask(user_message, chain)
        except Exception as e:
            bot_message = f"❌ **Error generating response:** {str(e)}"

    history = history or []
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_message})
    return "", history


def clear_history():
    return [], ""


# Custom CSS for modern styling
custom_css = """
#main-container { max-width: 1100px; margin: 0 auto; }
.header-box { text-align: center; margin-bottom: 20px; }
.header-box h1 { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 6px; }
.header-box p { font-size: 1.05rem; color: #64748B; }
.status-box { padding: 12px; border-radius: 8px; font-size: 0.95rem; }
"""

# Theme passed into gr.Blocks() instead of launch()
with gr.Blocks(title="RAG PDF Assistant", theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        # Header Section
        gr.HTML(
            """
            <div class="header-box">
                <h1>📚 AI RAG PDF Assistant</h1>
                <p>Upload or select a PDF document, ingest it, and ask questions powered by OpenAI & FAISS</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            # Left Sidebar: PDF Management & Ingestion
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Selection")

                existing_pdfs = list_pdfs()
                default_val = existing_pdfs[0] if existing_pdfs and existing_pdfs[0] != "No PDFs found" else None
                
                pdf_dropdown = gr.Dropdown(
                    choices=existing_pdfs,
                    value=default_val,
                    label="Select Existing PDF",
                    interactive=True
                )

                pdf_uploader = gr.File(
                    label="Or Upload New PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )

                ingest_btn = gr.Button("⚡ Ingest & Process PDF", variant="primary")

                ingest_status = gr.Markdown(
                    value="💡 Select or upload a PDF, then click **Ingest & Process PDF**.",
                    elem_classes=["status-box"]
                )

            # Right Main Panel: Chat Interface
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Document Q&A Chat")

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=450
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your question about the PDF document...",
                        label="Your Question",
                        scale=4,
                        lines=1
                    )
                    send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                    refresh_btn = gr.Button("🔄 Refresh PDF List", size="sm")

        # Event Handlers
        ingest_btn.click(
            fn=process_ingestion,
            inputs=[pdf_dropdown, pdf_uploader],
            outputs=[ingest_status, pdf_dropdown]
        )

        msg_input.submit(
            fn=bot_respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )

        send_btn.click(
            fn=bot_respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )

        clear_btn.click(
            fn=clear_history,
            outputs=[chatbot, msg_input]
        )

        refresh_btn.click(
            fn=lambda: gr.update(choices=list_pdfs()),
            outputs=[pdf_dropdown]
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )