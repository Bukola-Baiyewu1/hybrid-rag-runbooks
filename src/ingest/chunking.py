"""Step 2 of RAG: CHUNK each document into small, searchable pieces.

Why chunk at all? Two reasons:
  1. A whole document is too big to hand to the model as "context" for one
     question, and most of it is irrelevant to any single question anyway.
  2. Search works better on focused pieces. If someone asks "how do I reload the
     ingress?", we want to retrieve the few sentences about that, not a 40-page
     document.

A "chunk" is a small slice of text PLUS the metadata we need later: which file
it came from, which heading it sat under, how it was cut, the exact line range
it covers, and a checksum of its text. Source, heading, and lines are what let
the final answer cite its sources precisely.

Three strategies sit behind one interface, so the evaluation can compare them:
  * fixed     - fixed-size windows with overlap (the baseline)
  * headers   - one chunk per Markdown section (structure-aware)
  * semantic  - split where the topic shifts, measured with embeddings
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from .. import config
from .loaders import Document

_HEADING = re.compile(r"^\s*(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    text: str
    source: str  # filename it came from
    heading: str  # the section heading it sat under ("" if none)
    strategy: str  # which chunker produced it
    chunk_index: int  # its position within its document
    char_count: int
    start_line: int = 0  # 1-based, inclusive
    end_line: int = 0
    chunk_id: str = ""
    checksum: str = field(default="")

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    def citation(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "heading": self.heading,
            "lines": f"{self.start_line}-{self.end_line}",
        }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _stem(source: str) -> str:
    return source.rsplit(".", 1)[0]


def _line_of(text: str, offset: int) -> int:
    """1-based line number of a character offset."""
    return text.count("\n", 0, offset) + 1


def _heading_at(lines: list[str], line_no: int) -> str:
    """The nearest heading at or above a 1-based line number."""
    for i in range(min(line_no, len(lines)) - 1, -1, -1):
        m = _HEADING.match(lines[i])
        if m:
            return m.group(2)
    return ""


# --------------------------------------------------------------------------
# Strategy 1: fixed-size windows with overlap
# --------------------------------------------------------------------------
def chunk_fixed(doc: Document, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Slide a fixed-size window across the text, stepping forward by
    (size - overlap) each time. Simple and predictable — the baseline everyone
    compares against. We try to end each chunk on a space so we don't slice a
    word in half."""
    size = size or config.settings.chunk_size
    overlap = config.settings.chunk_overlap if overlap is None else overlap
    text = doc.text
    lines = text.splitlines()
    step = max(1, size - overlap)

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + size, len(text))
        # nudge the cut back to the last space, so words stay whole
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        raw = text[start:end]
        piece = raw.strip()
        if piece:
            first = start + (len(raw) - len(raw.lstrip()))
            last = first + len(piece)
            start_line, end_line = _line_of(text, first), _line_of(text, last - 1)
            chunks.append(
                Chunk(
                    piece,
                    doc.source,
                    _heading_at(lines, start_line),
                    "fixed",
                    index,
                    len(piece),
                    start_line,
                    end_line,
                    f"{_stem(doc.source)}#fixed-{index}",
                )
            )
            index += 1
        if end >= len(text):
            break
        start += step
    return chunks


# --------------------------------------------------------------------------
# Strategy 2: split on markdown headings (structure-aware)
# --------------------------------------------------------------------------
def chunk_by_headers(doc: Document, max_size: int | None = None) -> list[Chunk]:
    """Split the document at markdown headings (#, ##, ###...). Each section
    becomes a chunk that remembers its heading. This usually beats fixed-size on
    real docs because a section is already a coherent unit of meaning. If a
    section is longer than max_size, we fall back to fixed-size just for that
    section so no chunk is enormous.

    The heading line itself stays in the chunk text, so a search for the
    section's topic matches it, and chunk ids use the same "file#heading-slug"
    form as the Aegis agent, so citations line up between the two projects."""
    max_size = max_size or config.settings.max_section_size
    lines = doc.text.splitlines()
    title = next((m.group(2) for line in lines if (m := _HEADING.match(line)) and len(m.group(1)) == 1), "")

    sections: list[tuple[str, int, int]] = []  # heading, first line index, end index (exclusive)
    current_heading, current_start = "", 0
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if m:
            if i > current_start:
                sections.append((current_heading, current_start, i))
            current_heading, current_start = m.group(2), i
    sections.append((current_heading, current_start, len(lines)))

    chunks: list[Chunk] = []
    seen: set[str] = set()
    index = 0
    for heading, first, end in sections:
        while first < end and not lines[first].strip():
            first += 1
        while end > first and not lines[end - 1].strip():
            end -= 1
        body = "\n".join(lines[first:end]).strip()
        # skip sections that are only a heading line with nothing under it
        if not body or all(_HEADING.match(x) or not x.strip() for x in lines[first:end]):
            continue
        slug = "overview" if heading in ("", title) else _slug(heading)
        base_id = f"{_stem(doc.source)}#{slug}"
        if len(body) <= max_size:
            cid, n = base_id, 2
            while cid in seen:
                cid, n = f"{base_id}-{n}", n + 1
            seen.add(cid)
            chunks.append(Chunk(body, doc.source, heading, "headers", index, len(body), first + 1, end, cid))
            index += 1
            continue
        # section too big: reuse the fixed splitter, but keep the heading
        section_doc = Document("\n".join(lines[first:end]), doc.source)
        for part_no, sub in enumerate(chunk_fixed(section_doc, size=max_size), start=1):
            cid = f"{base_id}-part{part_no}"
            seen.add(cid)
            chunks.append(
                Chunk(
                    sub.text,
                    doc.source,
                    heading,
                    "headers",
                    index,
                    sub.char_count,
                    first + sub.start_line,
                    first + sub.end_line,
                    cid,
                )
            )
            index += 1
    return chunks


# --------------------------------------------------------------------------
# Strategy 3: semantic — split where the meaning shifts
# --------------------------------------------------------------------------
_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9`(\-])")


def _units(doc: Document) -> list[tuple[str, int]]:
    """Break a document into small units (sentences, list items, headings),
    each with its 1-based line number."""
    units: list[tuple[str, int]] = []
    paragraph: list[tuple[str, int]] = []

    def flush() -> None:
        if not paragraph:
            return
        text = " ".join(t for t, _ in paragraph)
        line = paragraph[0][1]
        for sentence in _SENTENCE_END.split(text):
            if sentence.strip():
                units.append((sentence.strip(), line))
        paragraph.clear()

    for i, raw in enumerate(doc.text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            flush()
        elif _HEADING.match(raw) or re.match(r"^(\d+\.|[-*])\s", line):
            flush()
            units.append((line, i))
        else:
            paragraph.append((line, i))
    flush()
    return units


def chunk_semantic(
    doc: Document,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
    percentile: float | None = None,
    max_size: int | None = None,
) -> list[Chunk]:
    """Embed each sentence, measure how different each sentence is from the one
    before it, and cut where that difference is in the top (100 - percentile)%.
    A heading always starts a new chunk, and no chunk grows past max_size."""
    from ..embed import cosine, get_embedder

    embed = embed or get_embedder().embed_passages
    percentile = config.settings.semantic_breakpoint_percentile if percentile is None else percentile
    max_size = max_size or config.settings.max_section_size
    lines = doc.text.splitlines()
    units = _units(doc)
    if not units:
        return []

    vectors = embed([t for t, _ in units])
    distances = [1.0 - cosine(vectors[i - 1], vectors[i]) for i in range(1, len(units))]
    threshold = sorted(distances)[int(len(distances) * percentile / 100)] if distances else 1.0

    groups: list[list[tuple[str, int]]] = [[units[0]]]
    for i in range(1, len(units)):
        text, _ = units[i]
        size = sum(len(t) + 1 for t, _ in groups[-1])
        is_heading = bool(_HEADING.match(text))
        after_heading = bool(_HEADING.match(units[i - 1][0]))  # never strand a heading alone
        shifted = distances[i - 1] >= threshold and not after_heading
        if is_heading or shifted or size + len(text) > max_size:
            groups.append([])
        groups[-1].append(units[i])

    chunks: list[Chunk] = []
    kept = [g for g in groups if not all(_HEADING.match(t) for t, _ in g)]
    for index, group in enumerate(kept):
        parts: list[str] = []
        for j, (text, line) in enumerate(group):
            same_line = j > 0 and group[j - 1][1] == line
            parts.append((" " if same_line else "\n") + text if j else text)
        text = "".join(parts).strip()
        start_line, end_line = group[0][1], group[-1][1]
        # a sentence may continue on wrapped lines: extend to the end of its paragraph
        while (
            end_line < len(lines)
            and lines[end_line].strip()
            and not _HEADING.match(lines[end_line])
            and not re.match(r"^\s*(\d+\.|[-*])\s", lines[end_line])
        ):
            end_line += 1
        chunks.append(
            Chunk(
                text,
                doc.source,
                _heading_at(lines, start_line),
                "semantic",
                index,
                len(text),
                start_line,
                end_line,
                f"{_stem(doc.source)}#semantic-{index}",
            )
        )
    return chunks


STRATEGIES: dict[str, Callable[..., list[Chunk]]] = {
    "fixed": chunk_fixed,
    "headers": chunk_by_headers,
    "semantic": chunk_semantic,
}


def chunk_document(doc: Document, strategy: str = "headers") -> list[Chunk]:
    """Chunk one document using the named strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES)}.")
    return STRATEGIES[strategy](doc)


def chunk_corpus(docs: list[Document], strategy: str = "headers") -> list[Chunk]:
    """Chunk a whole list of documents."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, strategy))
    return all_chunks
