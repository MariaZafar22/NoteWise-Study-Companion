"""Step 4 - Vector database: ChromaDB with cosine similarity, saved on disk."""
import chromadb

from . import config


class VectorStore:
    def __init__(self, path=None, collection=None):
        self.client = chromadb.PersistentClient(path=path or config.CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=collection or config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, records, embeddings):
        if not records:
            return 0
        self.collection.upsert(
            ids=[r["id"] for r in records],
            documents=[r["text"] for r in records],
            embeddings=embeddings,
            metadatas=[
                {"source": r["source"], "page": r["page"], "chunk_index": r["chunk_index"]}
                for r in records
            ],
        )
        return len(records)

    def search(self, query_embedding, top_k=4):
        count = self.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
        )
        hits = []
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - dist, 4),
            })
        return hits

    def sources(self):
        data = self.collection.get(include=["metadatas"])
        return sorted({m["source"] for m in data["metadatas"]})

    def count(self):
        return self.collection.count()

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
