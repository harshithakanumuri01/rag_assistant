# Multi-document RAG Research Assistant

Upload PDFs, PPTX, and DOCX files and ask questions across all of them.
Answers are grounded in your documents, with citations (filename + page/slide/section)
and multi-turn conversation memory.

## Core features

- **Multi-document upload** — PDF, PPTX, DOCX, any combination, at once
- **Semantic search** — FAISS + local `BAAI/bge-small-en-v1.5` embeddings (free, no API key, runs in-process)
- **Context-aware Q&A** — answers grounded only in retrieved excerpts, via Groq's free LLM API
- **Citations** — every answer shows source document + page/slide/section
- **Conversation memory** — follow-up questions use prior chat context

## Tech stack

| Layer | Tool |
|---|---|
| UI / hosting | Streamlit |
| Parsing | pdfplumber, python-docx, python-pptx |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`, local, free) |
| Vector store | FAISS (in-memory) |
| LLM | Groq API (`llama-3.3-70b-versatile`, free tier) |

## Project structure

```
rag-assistant/
├── app.py                  # Streamlit app (UI + orchestration)
├── requirements.txt
├── utils/
│   ├── document_parser.py  # PDF / DOCX / PPTX -> text + metadata
│   ├── chunking.py         # text -> overlapping chunks, metadata preserved
│   ├── vector_store.py     # embeddings + FAISS index + search
│   └── llm.py               # Groq call, prompt construction, citations
└── .streamlit/
    └── secrets.toml.example
```

## Run it locally

```bash
cd rag-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your free Groq API key (https://console.groq.com)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your real key

streamlit run app.py
```

The app opens at `http://localhost:8501`. The first run downloads the
embedding model (~130MB) from Hugging Face — needs internet access once,
then it's cached locally.

## Deploy for free on Streamlit Community Cloud

1. Push this folder to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app" → pick your repo, branch, and `app.py` as the entry point.
3. In **App settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your-real-key-here"
   ```
4. Deploy. Anyone with the link can use the app — they don't need their own API key, since yours is stored server-side in Secrets, never exposed to visitors.

> **Note:** Streamlit Community Cloud has no GPU and a shared CPU container.
> The embedding model runs fine (it's small), but very large PDFs (100+ pages)
> may take a little longer to index on first upload. This is normal.

## How it works (matches your workflow diagram)

1. **Upload** → file bytes read, dispatched by extension to the right parser
2. **Parse** → text extracted per page (PDF) / slide (PPTX) / section (DOCX), with metadata attached
3. **Chunk** → `RecursiveCharacterTextSplitter` (800 chars, 120 overlap) splits long pages into retrieval-sized pieces, metadata copied onto every chunk
4. **Embed + Index** → each chunk embedded with BGE-small, added to a FAISS `IndexFlatIP` (cosine similarity via normalized vectors)
5. **Query** → user question embedded the same way, top-5 most similar chunks retrieved
6. **Generate** → retrieved chunks + last 8 turns of chat history sent to Groq's Llama 3.3 model with a grounding system prompt
7. **Cite** → retrieved chunks' source/page/section shown in an expandable "Sources" panel under every answer
8. **Remember** → all Q&A pairs stored in `st.session_state.chat_history` for the session, feeding back into step 6 on the next question

## Extending later (per your full roadmap)

This is intentionally scoped to the **5 core features**. The modular structure
makes it straightforward to add, later:

- **Re-ranking** — add a `CrossEncoder` rerank step between `vector_store.search()` and `llm.generate_answer()` in `app.py`
- **AI agents** (summarize / compare / gap-analysis) — add new functions to `utils/llm.py` and new buttons/tabs in `app.py` that call them with all-document context instead of top-K retrieval
- **Persistent storage** — swap `VectorStore`'s in-memory FAISS index for a saved index on disk (or Pinecone/Chroma) so documents survive across sessions, and chat history into SQLite/Redis
- **Auth / multi-user** — Streamlit doesn't have built-in auth; would need `streamlit-authenticator` or a reverse proxy
