import os
import asyncio
import tempfile
import warnings
import hashlib
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# ---------------------------
# Async loop setup for Windows
# ---------------------------
if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

warnings.filterwarnings("ignore")

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Chat with your PDF", page_icon="🤖", layout="wide")
st.title("📄 Chat with your PDF")

# ---------------------------
# Load API key
# ---------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file.")
    st.stop()

# ---------------------------
# Helpers
# ---------------------------
def file_md5(file_bytes: bytes) -> str:
    """Return MD5 hash of file bytes."""
    m = hashlib.md5()
    m.update(file_bytes)
    return m.hexdigest()

@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_bytes: bytes, api_key: str, file_hash: str):
    """Build FAISS vector store from PDF bytes. Cached by file hash."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(pdf_bytes)
        temp_path = tf.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # embeddings endpoint
        google_api_key=api_key,
    )

    return FAISS.from_documents(chunks, embeddings)

def ensure_chain(vector_store, api_key: str, k_val: int):
    """Create chain with persistent memory."""
    if "pdf_memory" not in st.session_state:
        st.session_state.pdf_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": k_val})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.pdf_memory,
        return_source_documents=True,
        output_key="answer"
    )
    return chain

def render_sources(source_docs):
    with st.expander("📚 Source snippets"):
        for i, doc in enumerate(source_docs, start=1):
            meta = doc.metadata or {}
            page = meta.get("page", "N/A")
            st.markdown(f"**Source {i} (page {page}):**\n\n{doc.page_content}")

def safe_invoke_chain(chain, query: str):
    """Invoke chain with retry for 429 quota errors."""
    retries = 2
    for attempt in range(retries):
        try:
            return chain.invoke({"question": query})
        except Exception as e:
            if "429" in str(e):
                st.warning("⚠️ API quota hit. Retrying after delay...")
                time.sleep(60)  # backoff
            else:
                raise
    raise RuntimeError("API quota exceeded. Try later.")

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.subheader("Settings")
    k_val = st.slider("Top-K passages", 2, 8, 4, help="How many chunks to retrieve.")
    clear_chat = st.button("🧹 Clear chat history")

if clear_chat and "pdf_memory" in st.session_state:
    st.session_state.pdf_memory.clear()
    st.session_state.ui_messages = []

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is None:
    st.info("👆 Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded_file.read()
if not pdf_bytes:
    st.error("Uploaded file is empty.")
    st.stop()

pdf_hash = file_md5(pdf_bytes)

with st.spinner("🔎 Indexing your PDF..."):
    vector_store = build_vector_store(pdf_bytes, gemini_api_key, pdf_hash)

chain = ensure_chain(vector_store, gemini_api_key, k_val)

# ---------------------------
# Chat UI
# ---------------------------
st.divider()
st.subheader("Chat")

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# Display past messages
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about the PDF...")
if prompt:
    st.session_state.ui_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("💡 Thinking..."):
            try:
                response = safe_invoke_chain(chain, prompt)
            except Exception as e:
                st.error(f"❌ Error during chain invocation: {e}")
                st.stop()

        answer = response.get("answer") or response.get("result") or ""
        st.markdown(answer or "_No answer returned._")
        st.session_state.ui_messages.append({"role": "assistant", "content": answer})

        if "source_documents" in response and response["source_documents"]:
            render_sources(response["source_documents"])
