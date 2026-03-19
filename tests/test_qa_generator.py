"""Tests for synthetic QA generator — scaffolding tests, no live API calls.

TDD: tests verify prompt construction, chunk ID wiring, and dataset
assembly. The LLM call is mocked.

Citations:
  - _instructions.md L622 (synthetic QA generation)
  - _instructions.md L623 (per-config QA, chunk ID validation)
"""

from unittest.mock import patch, MagicMock

import pytest

from src.models import Chunk, ChunkMetadata, QAExample
from src.qa_generator import generate_qa_for_chunk, generate_qa_dataset


def _make_chunk(text: str, page: int, index: int) -> Chunk:
    """Helper to build a valid Chunk with minimal boilerplate."""
    return Chunk(
        text=text,
        page_number=page,
        chunk_index=index,
        start_char=0,
        end_char=len(text),
        method="fixed_size",
        metadata=ChunkMetadata(chunk_size=500, overlap=50, parser="pypdf"),
    )


@pytest.fixture
def sample_chunks():
    """A small set of chunks with distinct content for predictable QA."""
    return [
        _make_chunk("Revenue grew 15% year-over-year to $4.2 billion in fiscal 2022.", page=2, index=0),
        _make_chunk("The company expanded into three new international markets.", page=3, index=0),
        _make_chunk("Operating expenses decreased by 8% due to automation initiatives.", page=4, index=0),
    ]


def _make_mock_qa(chunk: Chunk) -> QAExample:
    """Helper: create a QAExample as if the LLM generated it."""
    return QAExample(
        question=f"What happened on page {chunk.page_number}?",
        relevant_chunk_ids=[chunk.id],
        source_page=chunk.page_number,
        chunk_method=chunk.method,
    )


class TestGenerateQAForChunk:
    """Verify single-chunk QA generation wiring."""

    def test_returns_qa_example(self, sample_chunks):
        chunk = sample_chunks[0]
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.return_value = "What was the revenue in fiscal 2022?"
            result = generate_qa_for_chunk(chunk)
        assert isinstance(result, QAExample)

    def test_chunk_id_in_relevant_ids(self, sample_chunks):
        """Generated QA must reference the source chunk's ID."""
        chunk = sample_chunks[0]
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.return_value = "What was the revenue growth?"
            result = generate_qa_for_chunk(chunk)
        assert chunk.id in result.relevant_chunk_ids

    def test_source_page_matches_chunk(self, sample_chunks):
        chunk = sample_chunks[1]
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.return_value = "Where did the company expand?"
            result = generate_qa_for_chunk(chunk)
        assert result.source_page == chunk.page_number

    def test_chunk_method_matches(self, sample_chunks):
        chunk = sample_chunks[0]
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.return_value = "What was the revenue?"
            result = generate_qa_for_chunk(chunk)
        assert result.chunk_method == chunk.method

    def test_chunk_text_included_in_prompt(self, sample_chunks):
        """The chunk's text must be passed to the LLM."""
        chunk = sample_chunks[0]
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.return_value = "Some question?"
            generate_qa_for_chunk(chunk)
            call_args = mock_llm.call_args
            # The chunk text should appear in the prompt passed to the LLM
            assert chunk.text in call_args[0][0]


class TestGenerateQADataset:
    """Verify dataset assembly from multiple chunks."""

    def test_generates_one_qa_per_chunk(self, sample_chunks):
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.side_effect = [
                "Question about revenue?",
                "Question about expansion?",
                "Question about expenses?",
            ]
            results = generate_qa_dataset(sample_chunks)
        assert len(results) == 3

    def test_each_qa_references_different_chunk(self, sample_chunks):
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.side_effect = [
                "Q1?", "Q2?", "Q3?",
            ]
            results = generate_qa_dataset(sample_chunks)
        all_chunk_ids = [qa.relevant_chunk_ids[0] for qa in results]
        assert len(set(all_chunk_ids)) == 3  # all unique

    def test_samples_n_chunks_when_specified(self, sample_chunks):
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.return_value = "Some question?"
            results = generate_qa_dataset(sample_chunks, n_samples=2)
        assert len(results) == 2

    def test_n_samples_capped_at_chunk_count(self, sample_chunks):
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.side_effect = ["Q1?", "Q2?", "Q3?"]
            results = generate_qa_dataset(sample_chunks, n_samples=100)
        assert len(results) == 3  # can't sample more than we have

    def test_skips_failed_generations(self, sample_chunks):
        """If LLM fails on one chunk, skip it and continue."""
        with patch("src.qa_generator._call_llm") as mock_llm:
            mock_llm.side_effect = [
                "Q1?",
                Exception("API error"),
                "Q3?",
            ]
            results = generate_qa_dataset(sample_chunks)
        assert len(results) == 2  # skipped the failed one
