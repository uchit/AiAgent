import logging
from typing import Iterable
import chromadb

logging.basicConfig(level=logging.INFO)

class MockEmbeddingFunction:
    """Simple mock embedding using ASCII codes."""
    def __init__(self, dim: int = 50):
        self.dim = dim

    def __call__(self, input: Iterable[str]):
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            text = str(text)
            vec = [float(ord(c))/1000 for c in text]
            if len(vec) < self.dim:
                vec += [0.0]*(self.dim - len(vec))
            else:
                vec = vec[:self.dim]
            embeddings.append(vec)
        return embeddings

    def embed_documents(self, input: Iterable[str]):
        return self.__call__(input)

    def embed_query(self, input: Iterable[str]):
        return self.__call__(input)

class VectorDB:
    def __init__(self, collection_name: str, persist_dir: str = ".chromadb"):
        logging.info("Initializing VectorDB...")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = MockEmbeddingFunction(dim=50)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
        )
        logging.info(f"Using collection '{collection_name}'.")

    def ingest(self, games: list):
        if not games:
            logging.warning("No valid games found to ingest.")
            return

        ids = [g["Name"] for g in games]
        documents = [g["Description"] for g in games]
        metadatas = [{k: g[k] for k in g if k != "Description"} for g in games]

        existing_ids = set(self.collection.get(include=[], ids=ids).get("ids", []))
        new_items = [
            (doc_id, doc, meta)
            for doc_id, doc, meta in zip(ids, documents, metadatas)
            if doc_id not in existing_ids
        ]

        if not new_items:
            logging.info("No new documents to add (all IDs already exist).")
            return

        new_ids = [item[0] for item in new_items]
        new_docs = [item[1] for item in new_items]
        new_meta = [item[2] for item in new_items]
        self.collection.add(ids=new_ids, documents=new_docs, metadatas=new_meta)
        logging.info("Ingestion complete!")

    def search(self, query: str, n_results: int = 3):
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )