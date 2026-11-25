"""The full pipeline: Document -> Chunking -> Embeddings -> Vector DB -> Retrieval -> LLM -> Answer."""
from pathlib import Path

from . import config, llm
from .chunker import chunk_pages
from .embeddings import Embedder
from .loader import load_document, load_folder
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(self, store_path=None, embedder=None):
        self.embedder = embedder or Embedder()
        self.store = VectorStore(path=store_path)

    # ---------- indexing ----------
    def ingest_pages(self, pages, chunk_size=None, overlap=None):
        records = chunk_pages(
            pages,
            chunk_size or config.CHUNK_SIZE,
            config.CHUNK_OVERLAP if overlap is None else overlap,
        )
        if not records:
            return 0
        vectors = self.embedder.embed([r["text"] for r in records])
        return self.store.add(records, vectors)

    def ingest_path(self, path, **kwargs):
        path = Path(path)
        pages = load_folder(path) if path.is_dir() else load_document(path)
        return self.ingest_pages(pages, **kwargs)

    # ---------- retrieval ----------
    def retrieve(self, question, top_k=None):
        vector = self.embedder.embed([question])[0]
        hits = self.store.search(vector, top_k or config.TOP_K)
        return [h for h in hits if h["score"] >= config.MIN_SCORE]

    # ---------- answering ----------
    def ask(self, question, top_k=None):
        hits = self.retrieve(question, top_k)
        if not hits:
            return {
                "answer": "I could not find anything relevant in the uploaded documents.",
                "sources": [],
                "mode": "no-context",
            }
        try:
            answer, mode = llm.generate_answer(question, hits), "llm"
        except Exception as error:  # no key, quota, network...
            answer = (
                "The language model is unavailable right now, so here are the most "
                f"relevant passages instead.\n\n{hits[0]['text']}"
            )
            mode = f"retrieval-only ({type(error).__name__})"
        return {"answer": answer, "sources": hits, "mode": mode}
