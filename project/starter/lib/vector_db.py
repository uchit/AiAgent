import os
import json
import logging
from typing import Iterable

try:
    import chromadb
except ImportError:  # pragma: no cover - dependency fallback
    chromadb = None

EMBED_CACHE_FILE = "embeddings_cache.json"
EMBEDDING_DIM = 50
LOGGER = logging.getLogger(__name__)


class MockEmbeddingFunction:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def name(self) -> str:
        return "mock-embedding-function"

    def __call__(self, input: Iterable[str]):
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            text = str(text)
            vec = [float(ord(c)) / 1000 for c in text]
            if len(vec) < self.dim:
                vec += [0.0] * (self.dim - len(vec))
            else:
                vec = vec[:self.dim]
            embeddings.append(vec)
        return embeddings

    def embed_documents(self, input: Iterable[str]):
        return self.__call__(input)

    def embed_query(self, input: Iterable[str]):
        return self.__call__(input)


class InMemoryCollection:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self._docs = {}

    def get(self, ids=None, include=None):
        if not ids:
            return {"ids": list(self._docs.keys())}
        found = [doc_id for doc_id in ids if doc_id in self._docs]
        return {"ids": found}

    def add(self, ids, documents, metadatas):
        embeddings = self.embedding_function(documents)
        for doc_id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            self._docs[doc_id] = {
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "embedding": emb,
            }

    def query(self, query_texts, n_results=5):
        query_embedding = self.embedding_function(query_texts)[0]
        scored = []
        for record in self._docs.values():
            distance = sum(
                (a - b) * (a - b)
                for a, b in zip(query_embedding, record["embedding"])
            )
            scored.append((distance, record))
        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]
        return {
            "ids": [[row["id"] for _, row in top]],
            "documents": [[row["document"] for _, row in top]],
            "metadatas": [[row["metadata"] for _, row in top]],
            "distances": [[dist for dist, _ in top]],
        }

class VectorDB:
    def __init__(self, collection_name="games_collection", persist_dir=".chromadb"):
        self.collection_name = collection_name
        self.embed_cache = self._load_cache()
        self.embed_fn = MockEmbeddingFunction()
        has_persistent_client = chromadb is not None and hasattr(chromadb, "PersistentClient")
        if not has_persistent_client:
            LOGGER.warning("chromadb is unavailable/incompatible; using in-memory fallback collection.")
            self.collection = InMemoryCollection(embedding_function=self.embed_fn)
        else:
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embed_fn,
            )

    def _load_cache(self) -> dict:
        if os.path.exists(EMBED_CACHE_FILE):
            with open(EMBED_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        with open(EMBED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.embed_cache, f)

    def _get_embedding(self, text: str):
        if text in self.embed_cache:
            return self.embed_cache[text]

        emb = self.embed_fn([text])[0]
        self.embed_cache[text] = emb
        self._save_cache()
        return emb

    def ingest(self, games: list[dict]) -> None:
        if not games:
            LOGGER.warning("No games provided for ingest.")
            return

        ids = [g["Name"] for g in games]
        docs = [g["Description"] for g in games]
        metas = [{k: v for k, v in g.items() if k != "Description"} for g in games]

        existing_ids = set(self.collection.get(ids=ids, include=[]).get("ids", []))
        new_rows = [
            (doc_id, doc, meta)
            for doc_id, doc, meta in zip(ids, docs, metas)
            if doc_id not in existing_ids
        ]
        if not new_rows:
            LOGGER.info("No new documents to ingest.")
            return

        self.collection.add(
            ids=[r[0] for r in new_rows],
            documents=[r[1] for r in new_rows],
            metadatas=[r[2] for r in new_rows],
        )

    def search(self, query: str, n_results: int = 5):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
