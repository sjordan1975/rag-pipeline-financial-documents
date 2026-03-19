"""
Text chunking strategies for the RAG pipeline.

Each chunker takes extracted text and returns a list of Chunk objects
with metadata. All chunkers share the same return type for uniformity
in the Phase 1 pre-grid.

Citations:
  - _instructions.md L602 (3 chunking strategies)
  - _instructions.md L604 (metadata: page number, method, size params, unique ID)
  - _instructions.md L605 (no mid-word/mid-sentence splits)
"""

import nltk
import spacy

nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize  # noqa: E402

from src.models import Chunk, ChunkMetadata

# Load spaCy model with word vectors for semantic chunking
_nlp = spacy.load("en_core_web_md")

# Cache spaCy docs to avoid re-parsing the same text across size/overlap combos
_spacy_cache: dict[int, tuple[list[str], list]] = {}


def _get_spacy_sentences(text: str) -> tuple[list[str], list]:
    """Parse text with spaCy once, cache and reuse across calls."""
    key = hash(text)
    if key in _spacy_cache:
        return _spacy_cache[key]

    doc = _nlp(text)
    sent_spans = [sent for sent in doc.sents if sent.text.strip()]
    sentences = [sent.text.strip() for sent in sent_spans]
    _spacy_cache[key] = (sentences, sent_spans)
    return sentences, sent_spans


def chunk_fixed_size(
    text: str,
    page_number: int,
    chunk_size: int,
    overlap: int,
    parser: str,
) -> list[Chunk]:
    """Split text into fixed-size chunks with word-boundary awareness.

    Args:
        text: The source text to chunk.
        page_number: Page number this text came from.
        chunk_size: Target chunk size in characters.
        overlap: Number of characters to overlap between consecutive chunks.
        parser: Name of the PDF parser used (for metadata).

    Returns:
        List of Chunk objects with metadata and character offsets.
    """
    if not text or not text.strip():
        return []

    metadata = ChunkMetadata(chunk_size=chunk_size, overlap=overlap, parser=parser)
    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take the rest
            end = len(text)
        else:
            # Back up to the nearest word boundary (space)
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end]

        # Skip chunks that are only whitespace
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text,
                page_number=page_number,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
                method="fixed_size",
                metadata=metadata,
            ))
            chunk_index += 1

        # Advance by (end - start - overlap), but at least 1 char to avoid infinite loop
        step = max(end - start - overlap, 1)
        start = start + step

    return chunks


def chunk_sentence(
    text: str,
    page_number: int,
    chunk_size: int,
    overlap: int,
    parser: str,
) -> list[Chunk]:
    """Split text into chunks at sentence boundaries using NLTK.

    Groups consecutive sentences until adding the next sentence would
    exceed chunk_size. Overlap is handled by repeating trailing sentences
    from the previous chunk.

    Args:
        text: The source text to chunk.
        page_number: Page number this text came from.
        chunk_size: Target chunk size in characters.
        overlap: Approximate character overlap — determines how many trailing
                 sentences from the previous chunk to repeat.
        parser: Name of the PDF parser used (for metadata).

    Returns:
        List of Chunk objects with metadata.
    """
    if not text or not text.strip():
        return []

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    metadata = ChunkMetadata(chunk_size=chunk_size, overlap=overlap, parser=parser)
    chunks: list[Chunk] = []
    chunk_index = 0

    i = 0  # index into sentences list
    while i < len(sentences):
        # Build a chunk by accumulating sentences
        current_sentences: list[str] = []
        current_len = 0

        while i < len(sentences):
            sent = sentences[i]
            new_len = current_len + len(sent) + (1 if current_sentences else 0)

            # If adding this sentence would exceed chunk_size and we already
            # have at least one sentence, stop here
            if current_sentences and new_len > chunk_size:
                break

            current_sentences.append(sent)
            current_len = new_len
            i += 1

        chunk_text = " ".join(current_sentences)

        # Find the chunk's position in the original text
        start_char = text.find(current_sentences[0])
        last_sent = current_sentences[-1]
        end_char = text.find(last_sent, start_char) + len(last_sent)

        chunks.append(Chunk(
            text=chunk_text,
            page_number=page_number,
            chunk_index=chunk_index,
            start_char=max(start_char, 0),
            end_char=max(end_char, 1),
            method="sentence",
            metadata=metadata,
        ))
        chunk_index += 1

        # Handle overlap: back up by enough sentences to cover ~overlap chars
        # but always advance at least 1 sentence to prevent infinite loops
        if overlap > 0 and i < len(sentences):
            overlap_chars = 0
            backtrack = 0
            max_backtrack = len(current_sentences) - 1  # keep at least 1 sentence of progress
            for j in range(len(current_sentences) - 1, -1, -1):
                if backtrack >= max_backtrack:
                    break
                overlap_chars += len(current_sentences[j])
                backtrack += 1
                if overlap_chars >= overlap:
                    break
            i = i - backtrack

    return chunks


def chunk_semantic(
    text: str,
    page_number: int,
    chunk_size: int,
    overlap: int,
    parser: str,
    similarity_threshold: float = 0.5,
) -> list[Chunk]:
    """Split text at semantic boundaries detected via spaCy word vectors.

    Algorithm:
    1. Split text into sentences using spaCy's sentence segmentation.
    2. Compute cosine similarity between consecutive sentence pairs.
    3. Mark a boundary where similarity drops below the threshold.
    4. Group sentences between boundaries into chunks, respecting chunk_size.

    Args:
        text: The source text to chunk.
        page_number: Page number this text came from.
        chunk_size: Target max chunk size in characters.
        overlap: Approximate character overlap (sentence-level backtrack).
        parser: Name of the PDF parser used (for metadata).
        similarity_threshold: Cosine similarity below which a boundary is placed.

    Returns:
        List of Chunk objects split at semantic boundaries.
    """
    if not text or not text.strip():
        return []

    # Single spaCy parse per unique text — cached across size/overlap combos
    sentences, sent_spans = _get_spacy_sentences(text)

    if not sent_spans:
        return []

    # Compute similarity between consecutive sentence spans and find boundaries
    # No second _nlp() call — spans already have vectors from the doc parse
    boundary_indices: list[int] = []
    for i in range(1, len(sent_spans)):
        if not sent_spans[i - 1].has_vector or not sent_spans[i].has_vector:
            continue
        sim = sent_spans[i - 1].similarity(sent_spans[i])
        if sim < similarity_threshold:
            boundary_indices.append(i)

    # Build chunks from sentence groups between boundaries
    metadata = ChunkMetadata(chunk_size=chunk_size, overlap=overlap, parser=parser)
    chunks: list[Chunk] = []
    chunk_index = 0

    # Add start and end as implicit boundaries
    all_boundaries = [0] + boundary_indices + [len(sentences)]

    i = 0
    while i < len(all_boundaries) - 1:
        start_sent = all_boundaries[i]
        end_sent = all_boundaries[i + 1]
        group = sentences[start_sent:end_sent]

        # If this group exceeds chunk_size, split further at chunk_size
        current_sentences: list[str] = []
        current_len = 0

        for sent in group:
            new_len = current_len + len(sent) + (1 if current_sentences else 0)

            if current_sentences and new_len > chunk_size:
                # Flush current chunk
                chunk_text = " ".join(current_sentences)
                start_char = text.find(current_sentences[0])
                last_sent = current_sentences[-1]
                end_char = text.find(last_sent, start_char) + len(last_sent)

                chunks.append(Chunk(
                    text=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    start_char=max(start_char, 0),
                    end_char=max(end_char, 1),
                    method="semantic",
                    metadata=metadata,
                ))
                chunk_index += 1
                current_sentences = []
                current_len = 0

            current_sentences.append(sent)
            current_len = current_len + len(sent) + (1 if len(current_sentences) > 1 else 0)

        # Flush remaining sentences in this group
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start_char = text.find(current_sentences[0])
            last_sent = current_sentences[-1]
            end_char = text.find(last_sent, start_char) + len(last_sent)

            chunks.append(Chunk(
                text=chunk_text,
                page_number=page_number,
                chunk_index=chunk_index,
                start_char=max(start_char, 0),
                end_char=max(end_char, 1),
                method="semantic",
                metadata=metadata,
            ))
            chunk_index += 1

        i += 1

    return chunks
