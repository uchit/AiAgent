import logging

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency fallback
    requests = None

logging.basicConfig(level=logging.INFO)

class TavilySearch:
    """Tavily API fallback search."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"  # Example

    def search(self, query: str, n_results: int = 3):
        if requests is None:
            logging.warning("requests is not installed; Tavily search is disabled.")
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"q": query, "limit": n_results}
        try:
            resp = requests.get(self.base_url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "url": item.get("url")
                })
            return results
        except Exception as e:
            logging.error(f"Tavily search error: {e}")
            return []

def evaluate_results(results, threshold: float = 0.35):
    """Return True if results are low-confidence and need web fallback."""
    if not results or not results.get("distances"):
        return True
    min_dist = min(results["distances"][0])
    return min_dist > threshold
