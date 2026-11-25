"""Step 2 - Text chunking with overlap, splitting on natural boundaries."""
import hashlib

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text, size, seps):
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep = seps[0]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split(text, size, seps[1:])
    pieces = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) > size:
            pieces.extend(_split(piece, size, seps[1:]))
        else:
            pieces.append(piece)
    return pieces


def split_text(text, chunk_size=800, overlap=150):
    """Split text into chunks of roughly chunk_size characters with overlap."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    pieces = _split(text.strip(), chunk_size, SEPARATORS)
    chunks, current = [], ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        tail = current[-overlap:] if overlap else ""
        if " " in tail:
            tail = tail[tail.index(" ") + 1:]
        current = tail + piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_pages(pages, chunk_size=800, overlap=150):
    """Turn loaded pages into chunk records ready for embedding."""
    records = []
    for page in pages:
        for index, text in enumerate(split_text(page["text"], chunk_size, overlap)):
            raw = f"{page['source']}|{page['page']}|{index}|{text[:60]}"
            records.append({
                "id": hashlib.sha1(raw.encode()).hexdigest(),
                "text": text,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": index,
            })
    return records
