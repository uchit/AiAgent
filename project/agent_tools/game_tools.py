import logging
from typing import List, Dict, Optional
try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

logging.basicConfig(level=logging.INFO)

# -----------------------------
# Tool 1: Internal Retrieval
# -----------------------------
class VectorDBTool:
    def __init__(self, vector_db):
        self.db = vector_db

    def retrieve(self, query: str, n_results: int = 3) -> List[Dict]:
        logging.info(f"[VectorDBTool] Retrieving for query: {query}")
        results = self.db.search(query, n_results=n_results)
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        items = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            items.append({
                "document": doc,
                "metadata": meta,
                "distance": dist
            })
        return items

# -----------------------------
# Tool 2: Result Evaluator
# -----------------------------
class ResultEvaluator:
    def __init__(self, distance_threshold: float = 0.35):
        self.threshold = distance_threshold

    def is_sufficient(self, results: List[Dict]) -> bool:
        if not results:
            logging.info("[ResultEvaluator] No results found.")
            return False
        for item in results:
            if item["distance"] < self.threshold:
                return True
        logging.info("[ResultEvaluator] Results not sufficient (all distances too high).")
        return False

# -----------------------------
# Tool 3: Web Search Fallback with Tavily
# -----------------------------
class WebSearchTool:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key or ""
        self.endpoint = "https://api.tavily.com/search"

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        if not self.api_key:
            logging.info("[WebSearchTool] No Tavily API key configured.")
            return []
        if requests is None:
            logging.warning("[WebSearchTool] requests is not installed.")
            return []
        logging.info(f"[WebSearchTool] Performing Tavily search for query: {query}")
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": n_results,
        }
        response = requests.post(self.endpoint, json=payload, timeout=20)
        if response.status_code != 200:
            logging.warning(f"[WebSearchTool] API call failed: {response.status_code} {response.text[:200]}")
            return []

        data = response.json()
        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("snippet")
            })
        return results
