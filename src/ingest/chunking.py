"""Step 2 of RAG: CHUNK each document into small, searchable pieces.

Why chunk at all? Two reasons:
  1. A whole document is too big to hand to the model as "context" for one
     question, and most of it is irrelevant to any single question anyway.
  2. Search works better on focused pieces. If someone asks "how do I reload the
     ingress?", we want to retrieve the few sentences about that, not a 40-page
     document.

A "chunk" is a small slice of text PLUS the metadata we need later: which file
it came from, which heading it sat under, how it was cut, and its position.
The source + heading are what let the final answer cite its sources.

We build two strategies now and leave a third (semantic) for the lesson where we
add embeddings, because semantic chunking needs them.
"""
from dataclasses import dataclass, asdict
from typing import List

from .. import config
from .loaders import Document


@dataclass
class Chunk:
    text: str
    source: str        # filename it came from
    heading: str       # the section heading it sat under ("" if none)
    strategy: str      # which chunker produced it
    chunk_index: int   # its position within its document
    char_count: int

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Strategy 1: fixed-size windows with overlap
# --------------------------------------------------------------------------
def chunk_fixed(doc: Document,
                size: int = None,
                overlap: int = None) -> List[Chunk]:
    """Slide a fixed-size window across the text, stepping forward by
    (size - overlap) each time. Simple and predictable — the baseline everyone
    compares against. We try to end each chunk on a space so we don't slice a
    word in half."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = doc.text
    step = max(1, size - overlap)

    chunks: List[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + size, len(text))
        # nudge the cut back to the last space, so words stay whole
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(piece, doc.source, "", "fixed", index, len(piece)))
            index += 1
        start += step
    return chunks


# --------------------------------------------------------------------------
# Strategy 2: split on markdown headings (structure-aware)
# --------------------------------------------------------------------------
def chunk_by_headers(doc: Document, max_size: int = None) -> List[Chunk]:
    """Split the document at markdown headings (#, ##, ###...). Each section
    becomes a chunk that remembers its heading. This usually beats fixed-size on
    real docs because a section is already a coherent unit of meaning. If a
    section is longer than max_size, we fall back to fixed-size just for that
    section so no chunk is enormous."""
    max_size = max_size or config.MAX_SECTION_SIZE
    lines = doc.text.splitlines()

    sections = []            # list of (heading, body_text)
    current_heading = ""
    current_body: List[str] = []

    def flush():
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_heading, body))

    for line in lines:
        if line.lstrip().startswith("#"):   # a markdown heading line
            flush()                          # close the previous section
            current_heading = line.lstrip("# ").strip()
            current_body = []
        else:
            current_body.append(line)
    flush()                                  # close the final section

    chunks: List[Chunk] = []
    index = 0
    for heading, body in sections:
        if len(body) <= max_size:
            chunks.append(Chunk(body, doc.source, heading, "headers", index, len(body)))
            index += 1
        else:
            # section too big: reuse the fixed splitter, but keep the heading
            for sub in chunk_fixed(Document(body, doc.source)):
                chunks.append(Chunk(sub.text, doc.source, heading, "headers", index, sub.char_count))
                index += 1
    return chunks


# --------------------------------------------------------------------------
# Strategy 3: semantic — added in the embeddings lesson
# --------------------------------------------------------------------------
def chunk_semantic(doc: Document) -> List[Chunk]:
    raise NotImplementedError(
        "Semantic chunking needs embeddings; we build it in the next lesson."
    )


STRATEGIES = {
    "fixed": chunk_fixed,
    "headers": chunk_by_headers,
    # "semantic": chunk_semantic,   # unlocked in the embeddings lesson
}


def chunk_document(doc: Document, strategy: str = "headers") -> List[Chunk]:
    """Chunk one document using the named strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES)}.")
    return STRATEGIES[strategy](doc)


def chunk_corpus(docs: List[Document], strategy: str = "headers") -> List[Chunk]:
    """Chunk a whole list of documents."""
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, strategy))
    return all_chunks
