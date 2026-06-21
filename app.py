"""
Multi-document RAG Research Assistant
Core 5 features: multi-doc upload, semantic search, context-aware QA,
citation generation, conversation memory.

Run locally:    streamlit run app.py
Deploy:         push to GitHub -> deploy on share.streamlit.io
                add GROQ_API_KEY in Streamlit Cloud's "Secrets" settings.
"""

import streamlit as st
from utils.document_parser import parse_file
from utils.chunking import chunk_records
from utils.vector_store import VectorStore
from utils.llm import generate_answer

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide",
)

TOP_K = 5  # number of chunks retrieved per query

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": "user"/"assistant", "content": str}]
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # filenames already indexed


def get_api_key() -> str | None:
    """
    Get the Groq API key from Streamlit secrets ONLY.
    There is intentionally no manual input fallback -- the key must live in
    Secrets (Streamlit Cloud: App settings -> Secrets, or locally in
    .streamlit/secrets.toml) so it's never typed into, displayed in, or
    sent from the visitor's browser.
    """
    if not hasattr(st, "secrets"):
        return None
    return st.secrets.get("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# Sidebar: upload + status + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Document Library")

    api_key = get_api_key()
    if not api_key:
        st.error(
            "App isn't configured yet. The site owner needs to add "
            "GROQ_API_KEY under App settings → Secrets in Streamlit Cloud."
        )

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs, DOCX, or PPTX",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} new file(s)..."):
                all_new_chunks = []
                for f in new_files:
                    try:
                        records = parse_file(f.read(), f.name)
                        chunks = chunk_records(records)
                        all_new_chunks.extend(chunks)
                        st.session_state.processed_files.add(f.name)
                    except Exception as e:
                        st.error(f"Failed to process {f.name}: {e}")
                if all_new_chunks:
                    st.session_state.vector_store.add(all_new_chunks)
            st.success(f"Indexed {len(new_files)} file(s)!")

    st.divider()
    if st.session_state.processed_files:
        st.markdown("**Indexed documents:**")
        for name in sorted(st.session_state.processed_files):
            st.markdown(f"- {name}")
        n_chunks = len(st.session_state.vector_store.chunks)
        st.caption(f"{n_chunks} chunks indexed total")
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    if st.button("Clear all documents & chat", use_container_width=True):
        st.session_state.vector_store = VectorStore()
        st.session_state.chat_history = []
        st.session_state.processed_files = set()
        st.rerun()

# ---------------------------------------------------------------------------
# Main panel: chat interface
# ---------------------------------------------------------------------------
st.title("Multi-document RAG Research Assistant")
st.caption("Upload your documents on the left, then ask questions across all of them.")

# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    loc = f"Page {s['page']}" if s.get("page") is not None else (s.get("section") or "N/A")
                    st.markdown(f"- **{s['source']}** ({loc}) — relevance: {s['score']:.2f}")

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    if st.session_state.vector_store.is_empty:
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"):
            st.warning("Please upload at least one document before asking a question.")
    elif not api_key:
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"):
            st.warning("This app isn't fully set up yet. Please check back later.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                retrieved = st.session_state.vector_store.search(question, top_k=TOP_K)
                try:
                    answer = generate_answer(
                        api_key=api_key,
                        question=question,
                        retrieved_chunks=retrieved,
                        chat_history=st.session_state.chat_history[:-1],  # exclude the just-added question
                    )
                except Exception as e:
                    answer = f"Sorry, I hit an error calling the LLM: {e}"

            st.markdown(answer)
            if retrieved:
                with st.expander("📎 Sources"):
                    for s in retrieved:
                        loc = f"Page {s['page']}" if s.get("page") is not None else (s.get("section") or "N/A")
                        st.markdown(f"- **{s['source']}** ({loc}) — relevance: {s['score']:.2f}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved,
        })
