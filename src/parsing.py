"""
PDF text extraction with multiple parser backends.

Each parser function has the same signature:
    parse_<lib>(pdf_path: str) -> list[tuple[int, str]]

Returns a list of (page_number, text) tuples, where page_number is 0-indexed.
This common interface lets the Phase 1 pre-grid iterate over parsers generically.

Citations:
  - _instructions.md L26 (parse PDFs using multiple extraction libraries)
  - _instructions.md L82 (pdfplumber, PyPDF2, PyMuPDF)
"""

from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from pypdf import PdfReader


def parse_pdfplumber(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text using pdfplumber — good at tables and structured layouts."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def parse_pypdf2(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text using PyPDF2 (via pypdf) — lightweight, fast."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def parse_pymupdf(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text using PyMuPDF (fitz) — fast, good general-purpose extraction."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        pages.append((i, text))
    doc.close()
    return pages


# Registry for programmatic iteration in Phase 1 pre-grid
PARSERS: dict[str, callable] = {
    "pdfplumber": parse_pdfplumber,
    "pypdf2": parse_pypdf2,
    "pymupdf": parse_pymupdf,
}
