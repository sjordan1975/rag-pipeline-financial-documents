"""
BM25 keyword retrieval — build index from text chunks, query top-K.

BM25 (Best Matching 25) is a bag-of-words ranking function that scores
documents by term frequency with diminishing returns and length normalization.
Complements vector (semantic) retrieval by catching exact keyword matches.

Citations:
  - _instructions.md L613 (BM25 keyword retrieval)
  - _instructions.md L614 (ranked results with scores)
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def build_bm25_index(
    texts: list[str],
    chunk_ids: list[str],
) -> tuple[BM25Okapi, dict[int, str]]:
    """Build a BM25 index from text chunks.

    Args:
        texts: List of chunk text strings.
        chunk_ids: List of chunk ID strings, same length as texts.

    Returns:
        Tuple of (BM25 index, position-to-chunk-ID mapping).
    """
    tokenized = [_tokenize(t) for t in texts]
    index = BM25Okapi(tokenized)
    id_map = {i: cid for i, cid in enumerate(chunk_ids)}
    return index, id_map


def query_bm25(
    index: BM25Okapi,
    id_map: dict[int, str],
    query: str,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Query the BM25 index for top-K results.

    Args:
        index: The BM25 index to search.
        id_map: Position-to-chunk-ID mapping from build_bm25_index.
        query: Query string.
        k: Number of results to return.

    Returns:
        List of (chunk_id, score) tuples, sorted by score descending.
    """
    tokenized_query = _tokenize(query)
    scores = index.get_scores(tokenized_query)

    # Pair each score with its position, sort descending
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # Clamp k to corpus size
    k = min(k, len(scored))

    return [(id_map[idx], float(score)) for idx, score in scored[:k]]
