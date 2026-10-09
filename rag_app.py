from __future__ import annotations

import hashlib
import json
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from rag_engine import (
    build_index,
    answer_question,
    get_document_stats,
    langsmith_status,
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "").strip()

st.set_page_config(
    page_title="Document Intelligence | RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --app-border: #e5e7eb; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
[data-testid="stSidebar"] { border-right: 1px solid var(--app-border); }
.hero { padding: 1.15rem 1.35rem; border: 1px solid var(--app-border);
        border-radius: 16px; margin-bottom: 1rem; }
.hero h1 { margin: 0 0 .35rem 0; letter-spacing: -.03em; }
.hero p { margin: 0; opacity: .78; }
div[data-testid="stMetric"] { border: 1px solid var(--app-border);
  border-radius: 12px; padding: .75rem; }
.source-card { border: 1px solid var(--app-border); border-radius: 10px;
  padding: .8rem; margin: .45rem 0; }
.small-muted { opacity: .72; font-size: .86rem; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
if "index_fingerprint" not in st.session_state:
    st.session_state.index_fingerprint = None
if "index_info" not in st.session_state:
    st.session_state.index_info = None
if "preparing_documents" not in st.session_state:
    st.session_state.preparing_documents = False
if "prepare_documents_requested" not in st.session_state:
    st.session_state.prepare_documents_requested = False

def _request_document_preparation():
    if not st.session_state.preparing_documents:
        st.session_state.preparing_documents = True
        st.session_state.prepare_documents_requested = True

st.markdown("""
<div class="hero">
  <h1>Document Intelligence</h1>
  <p>Find answers across your PDFs, with references to the source documents and pages.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    st.divider()
    st.subheader("Answer settings")
    model_name = st.selectbox(
        "Answer model",
        ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"],
        index=0,
        help="Model availability depends on your OpenAI account.",
    )
    retrieval_mode = st.selectbox(
        "Retrieval method",
        [
            "Hybrid search",
            "Basic semantic RAG",
            "BM25 keyword search",
            "Fusion RAG",
            "Hybrid + cross-encoder reranking",
            "Self-RAG",
            "Corrective RAG",
        ],
        index=0,
        help="Choose how relevant passages are found across your uploaded PDFs.",
    )
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 5
    st.caption("Five evidence passages are retrieved by default. Document chunking is managed automatically.")
    st.divider()
    tracing = langsmith_status()
    st.caption("Diagnostics: LangSmith tracing is configured." if tracing else "Diagnostics: LangSmith tracing is off.")

left, right = st.columns([1.55, 1], gap="large")
with left:
    st.subheader("Your documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload text-based PDFs. Scanned documents need OCR before they can be searched.",
    )
    if uploaded_files:
        total_bytes = sum(len(f.getvalue()) for f in uploaded_files)
        st.caption(f"{len(uploaded_files)} PDF(s) · {total_bytes / (1024 * 1024):.2f} MB")
        with st.expander("Files in this upload"):
            for f in uploaded_files:
                st.write(f"📄 {f.name} — {len(f.getvalue()) / 1024:.1f} KB")

    build_disabled = not uploaded_files or st.session_state.preparing_documents
    st.button(
        "Preparing documents…" if st.session_state.preparing_documents else "Prepare documents",
        type="primary",
        disabled=build_disabled,
        use_container_width=True,
        on_click=_request_document_preparation,
    )

    if st.session_state.prepare_documents_requested:
        st.session_state.prepare_documents_requested = False
        try:
            with st.status("Preparing documents and building the index…", expanded=True) as status:
                st.write("Extracting text and preparing searchable content.")
                built = build_index(
                    uploaded_files=uploaded_files,
                    embedding_model_name=embedding_model,
                )
                st.write("Building your document library.")
                st.session_state.rag_index = built
                fingerprint = hashlib.sha256(
                    b"".join(f.name.encode() + f.getvalue() for f in uploaded_files)
                    + f"{embedding_model}|900|150".encode()
                ).hexdigest()
                st.session_state.index_fingerprint = fingerprint
                st.session_state.index_info = get_document_stats(built)
                st.session_state.messages = []
                status.update(label="Documents ready", state="complete", expanded=False)
        except Exception:
            st.error("We could not prepare these documents. Check that the PDFs are readable and try again.")
            st.session_state.rag_index = None
            st.session_state.index_info = None
            st.session_state.index_fingerprint = None
        finally:
            st.session_state.preparing_documents = False
        st.rerun()

with right:
    st.subheader("Document library")
    info = st.session_state.index_info
    if info:
        m1, m2 = st.columns(2)
        m1.metric("PDF files", info["file_count"])
        m2.metric("Text chunks", info["chunk_count"])
        st.metric("Pages with extracted text", info["page_count"])
        for name in info["files"]:
            st.write(f"• {name}")
        st.caption("Your documents are available for this active session. Reloading the app clears them.")
        if st.button("Remove all documents", use_container_width=True):
            st.session_state.rag_index = None
            st.session_state.index_info = None
            st.session_state.index_fingerprint = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Upload PDFs to prepare your document library and start asking questions.")

st.divider()
st.subheader("Ask your documents")

if not api_key:
    st.warning("OpenAI API access is not configured. Add OPENAI_API_KEY to your local .env file, then restart the app to enable answers.")
if st.session_state.rag_index is None:
    st.info("Prepare your documents before asking a question.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Evidence used ({len(message['sources'])})"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**[{i}] {source['source']} — page {source['page']}**  \n"
                        f"<span class='small-muted'>{source.get('score_label', '')}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(source["content"])

question = st.chat_input("Ask a question about your documents…", disabled=(not api_key or st.session_state.rag_index is None))
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving evidence and preparing an answer…"):
                result = answer_question(
                    question=question,
                    index=st.session_state.rag_index,
                    mode=retrieval_mode,
                    model_name=model_name,
                    top_k=top_k,
                    api_key=api_key,
                )
            st.markdown(result["answer"])
            if result.get("sources"):
                with st.expander(f"Evidence used ({len(result['sources'])})"):
                    for i, source in enumerate(result["sources"], start=1):
                        st.markdown(f"**[{i}] {source['source']} — page {source['page']}**")
                        if source.get("score_label"):
                            st.caption(source["score_label"])
                        st.write(source["content"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "diagnostics": result.get("diagnostics", []),
            })
        except Exception as exc:
            error = "We could not generate an answer. Check your API account, model access and available credits, then try again."
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})

if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
