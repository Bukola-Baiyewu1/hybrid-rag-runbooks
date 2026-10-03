"""Step 1 of RAG: LOAD documents off disk into clean text.

A "loader" reads a file and hands back its text plus a note of where it came
from (the filename). We keep the source filename because later, when the system
answers a question, it must be able to say "this came from tls-cert-rotation.md".
That is what makes citations possible.

We support four file types. Markdown and plain text are read as-is. HTML and PDF
need a small helper library to pull the words out, and we only import those
libraries when we actually meet such a file — so you don't need them installed
to work with markdown runbooks.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .. import config


@dataclass
class Document:
    """One loaded file: its text, and the filename it came from."""
    text: str
    source: str   # e.g. "high-error-rate.md"


def _read_markdown_or_text(path: Path) -> str:
    # For markdown we deliberately keep the '#' headings in the text, because our
    # header-based chunker (next file) uses them to find section boundaries.
    return path.read_text(encoding="utf-8")


def _read_html(path: Path) -> str:
    from bs4 import BeautifulSoup  # imported only when we hit an .html file
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    return soup.get_text(separator="\n")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported only when we hit a .pdf file
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# Which file extensions we know how to read, and the function for each.
_READERS = {
    ".md": _read_markdown_or_text,
    ".txt": _read_markdown_or_text,
    ".html": _read_html,
    ".htm": _read_html,
    ".pdf": _read_pdf,
}


def load_file(path) -> Document:
    """Load a single file into a Document."""
    path = Path(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"I don't know how to read '{path.suffix}' files yet: {path}")
    text = reader(path).strip()
    return Document(text=text, source=path.name)


def load_corpus(corpus_dir: str = None) -> List[Document]:
    """Load every supported file in the corpus folder, sorted by name."""
    folder = Path(corpus_dir or config.CORPUS_DIR)
    if not folder.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {folder.resolve()}")

    docs: List[Document] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in _READERS:
            doc = load_file(path)
            if doc.text:                 # skip empty files
                docs.append(doc)
    return docs
