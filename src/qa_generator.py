"""
Synthetic QA generator — create evaluation questions from chunk text.

For each chunk, asks an LLM to generate a question that the chunk answers.
The chunk ID becomes the ground truth for retrieval evaluation. Uses
Instructor + Pydantic for structured output, same pattern as the
synthetic-data-generator project.

Citations:
  - _instructions.md L622 (synthetic QA generation)
  - _instructions.md L623 (per-config QA, chunk ID validation)
"""

from __future__ import annotations

import random

from src.models import Chunk, QAExample

# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Lazy-init Instructor-wrapped OpenAI client."""
    global _client
    if _client is not None:
        return _client

    import instructor
    from openai import OpenAI

    _client = instructor.from_openai(OpenAI())
    return _client


# ---------------------------------------------------------------------------
# LLM call (seam for mocking in tests)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are given a passage from a corporate annual report.
Generate a specific, factual question that can be answered using ONLY the information in this passage.

The question should:
- Be answerable from the passage alone
- Ask about a specific fact, number, or detail
- Be the kind of question a financial analyst or investor might ask

Passage:
{chunk_text}

Respond with ONLY the question, nothing else."""


def _call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call the LLM and return a question string.

    This is the seam that tests mock — keeps LLM interaction
    isolated from the wiring logic.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        response_model=None,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# QA generation
# ---------------------------------------------------------------------------


def generate_qa_for_chunk(
    chunk: Chunk,
    model: str = "gpt-4o-mini",
) -> QAExample:
    """Generate a QA pair from a single chunk.

    Args:
        chunk: The source chunk to generate a question from.
        model: LLM model to use for generation.

    Returns:
        A QAExample with the generated question and the chunk's ID as ground truth.
    """
    prompt = _PROMPT_TEMPLATE.format(chunk_text=chunk.text)
    question = _call_llm(prompt, model=model)

    return QAExample(
        question=question,
        relevant_chunk_ids=[chunk.id],
        source_page=chunk.page_number,
        chunk_method=chunk.method,
    )


def generate_qa_dataset(
    chunks: list[Chunk],
    n_samples: int | None = None,
    model: str = "gpt-4o-mini",
) -> list[QAExample]:
    """Generate QA pairs from a list of chunks.

    Args:
        chunks: All chunks from a chunking config.
        n_samples: Number of chunks to sample. None = use all.
        model: LLM model to use for generation.

    Returns:
        List of QAExample objects (may be fewer than n_samples if errors occur).
    """
    # Sample chunks if requested
    if n_samples is not None:
        n_samples = min(n_samples, len(chunks))
        selected = random.sample(chunks, n_samples)
    else:
        selected = chunks

    results: list[QAExample] = []
    for chunk in selected:
        try:
            qa = generate_qa_for_chunk(chunk, model=model)
            results.append(qa)
        except Exception:
            # Skip failed generations, continue with remaining chunks
            continue

    return results
