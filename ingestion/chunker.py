"""
ingestion/chunker.py
────────────────────────────────────────────────────────────────
Splits extracted PDF text into smart, retrievable chunks.

STRATEGY: Recursive Character Splitter
    Splits on natural boundaries in order:
      1. Paragraph breaks (\n\n)  ← try this first
      2. Line breaks (\n)
      3. Sentences (". ")
      4. Words (" ")              ← last resort, never cuts a word
    Only falls to next level if a piece is still too large.

WHY 600 TOKENS WITH 100 OVERLAP:
    TD Annual Report has dense financial paragraphs.
    600 tokens ≈ 3-4 paragraphs — one complete idea.
    100-token overlap ensures numbers/percentages that span
    a chunk boundary aren't lost.
────────────────────────────────────────────────────────────────
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List
from loguru import logger
import warnings
warnings.filterwarnings("ignore", category=Warning)

@dataclass
class Chunk:
    """
    One retrievable unit — the atomic building block of our RAG system.

    Every chunk carries enough metadata to show a citation like:
    "TD Annual Report 2025, Page 47"
    and to detect duplicates via content_hash.
    """
    chunk_id:     str
    doc_id:       str
    source_file:  str
    page_number:  int
    chunk_index:  int
    total_chunks: int
    text:         str
    token_count:  int
    content_hash: str
    metadata:     dict = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    """1 token ≈ 4 characters in English financial text. Fast approximation."""
    return len(text) // 4


def _hash_text(text: str) -> str:
    """16-char fingerprint for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _clean_text(text: str) -> str:
    """
    Light cleaning that preserves financial content.
    We keep numbers, percentages, and table structure intact.
    We only remove PDF artifacts and excessive whitespace.
    """
    # PDF artifacts
    text = text.replace('\x0c', '\n').replace('\x00', '')
    # Excessive spaces within a line
    text = re.sub(r'[ \t]{3,}', '  ', text)
    # More than 2 consecutive newlines → 2 (keeps paragraph structure)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Fix PDF hyphenation: "finan-\ncial" → "financial"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text.strip()


def _split_with_overlap(pieces: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Merge text pieces into chunks of target size, with overlap between them.

    Think of it like a sliding window:
    [----chunk 1----]
              [----chunk 2----]
    The overlapping part ensures context continuity.
    """
    chunks = []
    current = ""

    for piece in pieces:
        if _estimate_tokens(current + piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            # Overlap: take the tail of the previous chunk as context
            overlap_chars = chunk_overlap * 4
            overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else current
            current = overlap_text + piece

    if current.strip():
        chunks.append(current.strip())

    return chunks


# This is the most complex function. It tries to split text at natural language boundaries, from largest to smallest
def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text on progressively smaller boundaries until chunks fit.
    Tries paragraph → line → sentence → word (in that order).
    """
    separators = ["\n\n", "\n", ". ", " "]

    for sep in separators:
        if sep not in text:
            continue

        raw_pieces = text.split(sep)
        # Re-attach separator so text is not lost
        pieces = [p + sep for p in raw_pieces[:-1]] + [raw_pieces[-1]]

        # If any piece is still too large, recurse on that piece
        refined = []
        for piece in pieces:
            if _estimate_tokens(piece) > chunk_size:
                next_idx = separators.index(sep) + 1
                if next_idx < len(separators):
                    refined.extend(_recursive_split(piece, chunk_size, chunk_overlap))
                else:
                    refined.append(piece)  # give up, add as-is
            else:
                refined.append(piece)

        return _split_with_overlap(refined, chunk_size, chunk_overlap)

    return [text] if text.strip() else []


def chunk_document(
    pages: List[dict],
    doc_id: str,
    source_file: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Chunk]:
    """
    Convert extracted PDF pages → list of Chunk objects.

    Args:
        pages:        [{"page_number": int, "text": str}, ...]
        doc_id:       e.g. "td_annual_report_2025"
        source_file:  e.g. "td_annual_report_2025.pdf"
        chunk_size:   target tokens per chunk
        chunk_overlap: token overlap between consecutive chunks

    WHY PAGE-BY-PAGE:
        Financial reports cite by page ("See page 47").
        Processing page-by-page preserves page numbers for citations.
        If we concatenated all pages first, we'd lose this.
    """
    all_chunks: List[Chunk] = []
    seen_hashes: set = set()
    chunk_index = 0

    logger.info(
        f"Chunking '{source_file}' | pages={len(pages)} | "
        f"chunk_size={chunk_size} tokens | overlap={chunk_overlap} tokens"
    )

    for page in pages:
        page_num = page["page_number"]
        raw_text = page["text"]

        # Skip near-empty pages (table of contents, blank pages, cover)
        if not raw_text or len(raw_text.strip()) < 50:
            continue

        clean = _clean_text(raw_text)
        pieces = _recursive_split(clean, chunk_size, chunk_overlap)

        for piece in pieces:
            if len(piece.strip()) < 50:
                continue  # too short to be useful

            content_hash = _hash_text(piece)
            if content_hash in seen_hashes:
                continue  # exact duplicate — skip
            seen_hashes.add(content_hash)

            chunk = Chunk(
                chunk_id=f"{doc_id}_p{page_num:03d}_c{chunk_index:04d}",
                doc_id=doc_id,
                source_file=source_file,
                page_number=page_num,
                chunk_index=chunk_index,
                total_chunks=0,  # filled after loop
                text=piece,
                token_count=_estimate_tokens(piece),
                content_hash=content_hash,
                metadata={
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "token_count": _estimate_tokens(piece),
                }
            )
            all_chunks.append(chunk)
            chunk_index += 1

    # Fill in total now that we know it
    total = len(all_chunks)
    for chunk in all_chunks:
        chunk.total_chunks = total
        chunk.metadata["total_chunks"] = total

    logger.success(f"'{source_file}' → {total} chunks created")
    return all_chunks
