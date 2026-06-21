"""
Vector store utilities.
Wraps a FAISS index + the BGE local embedding model behind a simple
class so the rest of the app doesn't need to know about either.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class VectorStore:
    """An in-memory FAISS index paired with chunk metadata."""

    def __init__(self):
        self._model = None  # lazy-loaded, see `model` property
        self.index = None
        self.chunks: list[dict] = []  # parallel array: chunks[i] <-> index vector i

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        # BGE models recommend a query instruction prefix for queries (not for documents)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,  # so we can use inner product as cosine similarity
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def build(self, chunks: list[dict]):
        """Build (or rebuild) the FAISS index from a list of chunk dicts."""
        if not chunks:
            self.index = None
            self.chunks = []
            return

        self.chunks = chunks
        vectors = self._embed([c["text"] for c in chunks])
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
        index.add(vectors)
        self.index = index

    def add(self, chunks: list[dict]):
        """Add more chunks to an existing index (used when uploading additional docs)."""
        if self.index is None:
            self.build(chunks)
            return
        if not chunks:
            return
        vectors = self._embed([c["text"] for c in chunks])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k most relevant chunks for a query, with similarity scores."""
        if self.index is None or len(self.chunks) == 0:
            return []
        query_vector = self._embed([query])
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    @property
    def is_empty(self) -> bool:
        return self.index is None or len(self.chunks) == 0
