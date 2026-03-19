"""
Vector retrieval — text-in, ranked-results-out wrapper around FAISS.

Bridges embedding.py and vector_store.py: takes a text query, embeds it,
searches the FAISS index, and returns ranked (chunk_id, distance) tuples.
Same output interface as bm25_retrieval for downstream interchangeability.

Citations:
  - _instructions.md L611 (FAISS stores and retrieves correctly)
  - _instructions.md L612 (top-K with configurable K)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.embedding import embed_texts
from src.vector_store import build_index, query_index


def build_vector_retriever(
    embeddings: np.ndarray,
    chunk_ids: list[str],
) -> dict[str, Any]:
    """Build a vector retriever from pre-computed embeddings.

    Args:
        embeddings: numpy array of shape (n_chunks, dimension).
        chunk_ids: list of chunk ID strings.

    Returns:
        Retriever dict with 'index', 'id_map', and 'embeddings' keys.
    """
    index, id_map = build_index(embeddings, chunk_ids)
    return {
        "index": index,
        "id_map": id_map,
        "embeddings": embeddings,
    }


def query_vector(
    retriever: dict[str, Any],
    query: str,
    model: str = "text-embedding-3-small",
    k: int = 5,
) -> list[tuple[str, float]]:
    """Query with text: embed the query, then search FAISS.

    Args:
        retriever: Retriever dict from build_vector_retriever.
        query: Raw text query string.
        model: Embedding model to use for the query.
        k: Number of results to return.

    Returns:
        List of (chunk_id, distance) tuples, sorted by distance ascending.
    """
    query_embedding = embed_texts([query], model=model)
    return query_index(retriever["index"], retriever["id_map"], query_embedding, k=k)
