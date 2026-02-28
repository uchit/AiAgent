import logging
import os
import re
from typing import Iterable, Optional
try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

logging.basicConfig(level=logging.INFO)

MOCK_EMBEDDING_DIM = 50
OPENAI_EMBEDDING_DIM = 1536


class MockEmbeddingFunction:
    """Simple mock embedding using ASCII codes."""
    def __init__(self, dim: int = MOCK_EMBEDDING_DIM):
        self.dim = dim

    def name(self) -> str:
        return "mock-embedding-function"

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


class OpenAIEmbeddingFunction:
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def name(self) -> str:
        return f"openai-{self.model}"

    def __call__(self, input: Iterable[str]):
        if isinstance(input, str):
            input = [input]
        text_list = [str(item) for item in input]
        response = self.client.embeddings.create(model=self.model, input=text_list)
        return [item.embedding for item in response.data]

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

    def query(self, query_texts, n_results=3, include=None):
        query_embedding = self.embedding_function(query_texts)[0]
        scored = []
        for record in self._docs.values():
            distance = sum((a - b) * (a - b) for a, b in zip(query_embedding, record["embedding"]))
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
    def __init__(
        self,
        collection_name: str,
        persist_dir: str = ".chromadb",
        openai_model: str = "text-embedding-3-small",
    ):
        logging.info("Initializing VectorDB...")
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._supports_reset = False
        self._records = {}

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.embed_fn = OpenAIEmbeddingFunction(model=openai_model, api_key=api_key)
                self.embedding_mode = "openai"
                self.embedding_dim = OPENAI_EMBEDDING_DIM
                logging.info("Using OpenAI embeddings (%s).", openai_model)
            except Exception as exc:
                logging.warning("OpenAI embeddings unavailable (%s); using mock embeddings.", exc)
                self.embed_fn = MockEmbeddingFunction(dim=MOCK_EMBEDDING_DIM)
                self.embedding_mode = "mock"
                self.embedding_dim = MOCK_EMBEDDING_DIM
        else:
            self.embed_fn = MockEmbeddingFunction(dim=MOCK_EMBEDDING_DIM)
            self.embedding_mode = "mock"
            self.embedding_dim = MOCK_EMBEDDING_DIM
            logging.info("OPENAI_API_KEY not set; using mock embeddings.")

        has_persistent_client = chromadb is not None and hasattr(chromadb, "PersistentClient")
        if has_persistent_client:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn,
            )
            self._supports_reset = True
        else:
            logging.warning("chromadb is unavailable/incompatible; using in-memory fallback collection.")
            self.collection = InMemoryCollection(embedding_function=self.embed_fn)
        logging.info("Using collection '%s'.", self.collection_name)

    def _fallback_to_mock_embeddings(self):
        if self.embedding_mode == "mock":
            return
        logging.warning("Falling back to mock embeddings due to OpenAI embedding failure.")
        self.embed_fn = MockEmbeddingFunction(dim=MOCK_EMBEDDING_DIM)
        self.embedding_mode = "mock"
        self.embedding_dim = MOCK_EMBEDDING_DIM
        if self._supports_reset:
            self._reset_collection()
        else:
            self.collection = InMemoryCollection(embedding_function=self.embed_fn)

    def _reset_collection(self):
        if not self._supports_reset:
            return
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
        )

    def _run_with_dimension_recovery(self, func, op_name: str):
        try:
            return func()
        except Exception as exc:
            message = str(exc).lower()
            if any(token in message for token in ("insufficient_quota", "rate limit", "error code: 429")):
                self._fallback_to_mock_embeddings()
                return func()
            if self._supports_reset and "dimension" in message:
                logging.warning(
                    "Embedding dimension mismatch during %s. Resetting collection '%s' for %s embeddings.",
                    op_name,
                    self.collection_name,
                    self.embedding_mode,
                )
                self._reset_collection()
                return func()
            raise

    def _normalize_tokens(self, text: str):
        return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]

    def _build_local_records(self, games: list):
        for game in games:
            game_id = game["Name"]
            self._records[game_id] = {
                "id": game_id,
                "document": game["Description"],
                "metadata": {k: v for k, v in game.items() if k != "Description"},
            }

    def _lexical_search(self, query: str, n_results: int = 3):
        if not self._records:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q_tokens = set(self._normalize_tokens(query))
        q_norm = " ".join(self._normalize_tokens(query))
        scored = []
        for record in self._records.values():
            meta = record["metadata"]
            name = str(meta.get("Name", ""))
            document = record["document"]
            haystack = f"{name} {meta.get('Platform', '')} {meta.get('Genre', '')} {meta.get('Publisher', '')} {document}"
            h_tokens = set(self._normalize_tokens(haystack))
            overlap = len(q_tokens & h_tokens)
            base = overlap / max(1, len(q_tokens))

            phrase_bonus = 0.0
            name_norm = " ".join(self._normalize_tokens(name))
            if name_norm and name_norm in q_norm:
                phrase_bonus += 0.6
            if any(tok in name_norm for tok in q_tokens):
                phrase_bonus += 0.2

            score = min(1.0, base + phrase_bonus)
            distance = 1.0 - score
            scored.append((distance, record))

        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]
        return {
            "ids": [[row["id"] for _, row in top]],
            "documents": [[row["document"] for _, row in top]],
            "metadatas": [[row["metadata"] for _, row in top]],
            "distances": [[dist for dist, _ in top]],
        }

    def ingest(self, games: list):
        if not games:
            logging.warning("No valid games found to ingest.")
            return
        self._build_local_records(games)

        ids = [g["Name"] for g in games]
        documents = [g["Description"] for g in games]
        metadatas = [{k: g[k] for k in g if k != "Description"} for g in games]

        def _ingest():
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

        self._run_with_dimension_recovery(_ingest, "ingest")

    def search(self, query: str, n_results: int = 3):
        if self.embedding_mode == "mock":
            return self._lexical_search(query=query, n_results=n_results)

        def _query():
            return self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
            )

        return self._run_with_dimension_recovery(_query, "query")
