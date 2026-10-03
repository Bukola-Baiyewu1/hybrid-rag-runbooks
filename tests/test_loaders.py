import pytest

from src.ingest.loaders import load_corpus, load_file


def test_html_loader_extracts_text(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><body><h1>Restart</h1><p>Use <b>restart_service</b> carefully.</p></body></html>")
    doc = load_file(p)
    assert "restart_service" in doc.text and "<b>" not in doc.text
    assert doc.source == "page.html"


def _tiny_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_pdf_loader_extracts_text(tmp_path):
    p = tmp_path / "runbook.pdf"
    p.write_bytes(_tiny_pdf("Rotate the database password every 90 days"))
    assert "every 90 days" in load_file(p).text


def test_unknown_extension_is_rejected(tmp_path):
    p = tmp_path / "notes.docx"
    p.write_text("x")
    with pytest.raises(ValueError):
        load_file(p)


def test_missing_corpus_folder_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(str(tmp_path / "nope"))


def test_empty_files_are_skipped(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\ncontent")
    (tmp_path / "b.md").write_text("   \n")
    assert [d.source for d in load_corpus(str(tmp_path))] == ["a.md"]
