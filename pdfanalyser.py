import os
import asyncio
import tempfile
import warnings
import hashlib
import time
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# ---------------------------
# Async loop setup for Windows
# ---------------------------
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Chat with your PDF", page_icon="🤖", layout="wide")

# ---------------------------
# API Key Input & Validation
# ---------------------------
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "api_validated" not in st.session_state:
    st.session_state.api_validated = False

def validate_api_key(api_key: str) -> bool:
    """Validate the Gemini API key by making a small test request."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0,
        )
        _ = llm.invoke("ping")
        return True
    except Exception as e:
        st.error(f"❌ API Key validation failed: {e}")
        return False

# STEP 1 — Block until API key is validated
if not st.session_state.api_validated:
    st.title("🔑 Enter Gemini API Key")
    api_key_input = st.text_input(
        "Please enter your Gemini API Key:",
        type="password",
        placeholder="AI....",
        help="Your key will only be stored for this session."
    )

    if st.button("Continue"):
        if api_key_input.strip():
            if validate_api_key(api_key_input.strip()):
                st.session_state.gemini_api_key = api_key_input.strip()
                st.session_state.api_validated = True
                st.success("✅ API Key validated successfully!")
                st.rerun()
            else:
                st.error("Invalid API Key. Please try again.")
        else:
            st.error("API key cannot be empty.")
    st.stop()

# ---------------------------
# Helpers
# ---------------------------
def file_md5(file_bytes: bytes) -> str:
    m = hashlib.md5()
    m.update(file_bytes)
    return m.hexdigest()

@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_bytes: bytes, api_key: str, file_hash: str):
    # Ensure an event loop exists (important for Windows + Streamlit)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

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
        model="models/embedding-001",
        google_api_key=api_key,
    )
    return FAISS.from_documents(chunks, embeddings)

def ensure_chain(vector_store, api_key: str, k_val: int):
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

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.pdf_memory,
        return_source_documents=True,
        output_key="answer"
    )

def render_sources(source_docs):
    with st.expander("📚 Source snippets"):
        for i, doc in enumerate(source_docs, start=1):
            page = doc.metadata.get("page", "N/A")
            st.markdown(f"**Source {i} (page {page}):**\n\n{doc.page_content}")

def safe_invoke_chain(chain, query: str):
    retries = 2
    for _ in range(retries):
        try:
            return chain.invoke({"question": query})
        except Exception as e:
            if "429" in str(e):
                st.warning("⚠️ API quota hit. Retrying after delay...")
                time.sleep(60)
            else:
                raise
    raise RuntimeError("API quota exceeded. Try later.")

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.subheader("Settings")
    k_val = st.slider("Top-K passages", 2, 8, 4)
    if st.button("🧹 Clear chat history"):
        if "pdf_memory" in st.session_state:
            st.session_state.pdf_memory.clear()
        st.session_state.ui_messages = []

# ---------------------------
# Main App
# ---------------------------
st.title("📄 Chat with your PDF")

gemini_api_key = st.session_state.gemini_api_key
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

st.divider()
st.subheader("Chat")

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

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
                st.error(f"❌ Error: {e}")
                st.stop()

        answer = response.get("answer") or response.get("result") or ""
        st.markdown(answer or "_No answer returned._")
        st.session_state.ui_messages.append({"role": "assistant", "content": answer})

        if "source_documents" in response and response["source_documents"]:
            render_sources(response["source_documents"])

