"""Deterministic text chunking with character-level overlap."""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, *, size: int = 800, overlap: int = 120) -> list[str]:
    """Split `text` into overlapping chunks of roughly `size` characters.

    Prefers sentence boundaries when possible so chunks don't split mid-sentence.
    Falls back to hard character slicing for very long unbroken runs.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")

    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sentences = _SENTENCE_BOUNDARY.split(text)
    chunks: list[str] = []
    buf = ""

    for sentence in sentences:
        if not sentence:
            continue
        prospective = (buf + " " + sentence).strip() if buf else sentence
        if len(prospective) <= size:
            buf = prospective
            continue

        if buf:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + " " + sentence).strip() if tail else sentence
        else:
            # Single sentence longer than size — hard-slice
            for i in range(0, len(sentence), size - overlap):
                chunks.append(sentence[i : i + size])
            buf = ""

    if buf:
        chunks.append(buf)

    return chunks
