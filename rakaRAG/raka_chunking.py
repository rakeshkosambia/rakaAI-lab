"""Chunking strategies for rakaRAG. Pure standard library, original code.

Every chunk keeps its character span in the source document, so a chunk can always be traced back
to the exact text it came from (a citation later needs this). Sizes are counted in words.

Strategies:
  doc        the whole document as one chunk (the r01 behaviour)
  fixed      a window of `size` words that slides by `size - overlap` words
  recursive  split on the biggest natural boundary first (blank line, line, sentence, clause, word),
             then merge neighbouring pieces back together up to `size` words
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import raka_common as rc

STRATEGIES = ("doc", "fixed", "recursive")
SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str
    start: int  # character offset in the source document (inclusive)
    end: int    # character offset in the source document (exclusive)

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"

    @property
    def words(self) -> int:
        return len(self.text.split())


def _words(text: str) -> int:
    return len(text.split())


def check_sizes(size: int, overlap: int) -> None:
    """Same rule as LightRAG's token chunker: a window must move forward."""
    if size < 1:
        raise ValueError(f"size must be at least 1 word, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")


def _make(doc: rc.Doc, spans: list[tuple[int, int]]) -> list[Chunk]:
    """Turn raw spans into numbered chunks: trim whitespace, drop empties, keep exact offsets."""
    chunks: list[Chunk] = []
    for start, end in spans:
        piece = doc.text[start:end]
        stripped = piece.strip()
        if not stripped:
            continue
        lead = len(piece) - len(piece.lstrip())
        chunks.append(Chunk(doc.id, len(chunks), stripped, start + lead, start + lead + len(stripped)))
    return chunks


def chunk_whole(doc: rc.Doc) -> list[Chunk]:
    return _make(doc, [(0, len(doc.text))])


def chunk_fixed(doc: rc.Doc, size: int, overlap: int) -> list[Chunk]:
    check_sizes(size, overlap)
    words = list(_WORD.finditer(doc.text))
    spans = []
    step = size - overlap
    for first in range(0, len(words), step):
        last = min(first + size, len(words))
        spans.append((words[first].start(), words[last - 1].end()))
        if last == len(words):
            break  # the final window reached the end; do not emit a tail that only repeats overlap
    return _make(doc, spans)


def _split_spans(text: str, offset: int, sep: str) -> list[tuple[int, int]]:
    """Cut text after each occurrence of sep (the separator stays with the piece before it)."""
    spans, pos = [], 0
    while True:
        i = text.find(sep, pos)
        if i < 0:
            spans.append((pos, len(text)))
            break
        spans.append((pos, i + len(sep)))
        pos = i + len(sep)
    return [(offset + a, offset + b) for a, b in spans if text[a:b].strip()]


def _recurse(full: str, start: int, end: int, size: int, seps: tuple[str, ...]) -> list[tuple[int, int]]:
    text = full[start:end]
    if _words(text) <= size:
        return [(start, end)]
    sep = next((s for s in seps if s in text), None)
    if sep is None:
        return [(start, end)]
    rest = seps[seps.index(sep) + 1:]
    pieces: list[tuple[int, int]] = []
    for a, b in _split_spans(text, start, sep):
        pieces.extend(_recurse(full, a, b, size, rest))
    merged: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for a, b in pieces:
        if current and _words(full[current[0]:b]) <= size:
            current = (current[0], b)
        else:
            if current:
                merged.append(current)
            current = (a, b)
    if current:
        merged.append(current)
    return merged


def chunk_recursive(doc: rc.Doc, size: int) -> list[Chunk]:
    check_sizes(size, 0)
    return _make(doc, _recurse(doc.text, 0, len(doc.text), size, SEPARATORS))


def chunk_doc(doc: rc.Doc, strategy: str, size: int = 40, overlap: int = 0) -> list[Chunk]:
    if strategy == "doc":
        return chunk_whole(doc)
    if strategy == "fixed":
        return chunk_fixed(doc, size, overlap)
    if strategy == "recursive":
        return chunk_recursive(doc, size)
    raise ValueError(f"unknown strategy {strategy!r}; choose one of {STRATEGIES}")


def chunk_corpus(docs: list[rc.Doc], strategy: str, size: int = 40, overlap: int = 0) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_doc(doc, strategy, size, overlap))
    return out


def as_docs(chunks: list[Chunk]) -> list[rc.Doc]:
    """View chunks as index-able documents (id = 'doc#n') so the r01 indexes can search them."""
    return [rc.Doc(c.id, c.doc_id, c.text) for c in chunks]
