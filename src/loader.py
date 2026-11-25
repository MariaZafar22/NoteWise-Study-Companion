"""Step 1 - Document processing: read PDF / TXT / MD files into raw pages."""
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


def load_document(path, display_name=None):
    """Return a list of {"text", "source", "page"} dicts, one per page."""
    path = Path(path)
    ext = path.suffix.lower()
    source = display_name or path.name
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")

    pages = []
    if ext == ".pdf":
        reader = PdfReader(str(path))
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "source": source, "page": number})
    else:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            pages.append({"text": text, "source": source, "page": 1})
    return pages


def load_folder(folder):
    """Load every supported file inside a folder."""
    pages = []
    for file in sorted(Path(folder).iterdir()):
        if file.suffix.lower() in SUPPORTED:
            pages.extend(load_document(file))
    return pages
