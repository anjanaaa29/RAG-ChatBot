import streamlit as st
import requests
import os
from streamlit_chat import message

# API endpoints
API_URL_CHAT = os.getenv("API_URL_CHAT", "http://127.0.0.1:8000/query")
API_URL_PROCESS = os.getenv("API_URL_PROCESS", "http://127.0.0.1:8000/process-documents")

st.set_page_config(page_title="📚 RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []  # [(user, bot)]

# --- Sidebar: Upload & Clear ---
st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Select files to upload and index",
    type=["pdf", "docx", "txt", "csv", "json"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please select at least one file.")
    else:
        saved_paths = []
        temp_dir = "uploaded_files"
        os.makedirs(temp_dir, exist_ok=True)

        for file in uploaded_files:
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(path)

        with st.spinner("Processing documents..."):
            try:
                resp = requests.post(API_URL_PROCESS, json={"file_paths": saved_paths})
                resp.raise_for_status()
                data = resp.json()
                st.sidebar.success(f"✅ {data['document_count']} documents indexed.")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.sidebar.info("Chat history cleared.")

st.markdown("---")

# --- Chat Section ---
st.header("💬 Chat with your documents")

query = st.chat_input("Ask a question about the documents…")

if query:
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(API_URL_CHAT, json={"question": query})
            resp.raise_for_status()
            data = resp.json()
            st.session_state.history.append(
                (query, data["answer"])
            )
        except Exception as e:
            st.error(f"Error: {e}")

# --- Display Chat History ---
if not st.session_state.history:
    st.info("💬 Start by asking a question!")
else:
    for idx, (q, a) in enumerate(st.session_state.history):
        message(q, is_user=True, key=f"user_{idx}")
        message(a, key=f"bot_{idx}")
