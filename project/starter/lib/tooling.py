from tavily import TavilyClient
import os


class GameWebSearch:
    def __init__(self):
        self.client = TavilyClient(
            api_key=os.getenv("TAVILY_API_KEY")
        )

    def search(self, query: str) -> str:
        response = self.client.search(
            query=query,
            search_depth="advanced",
            max_results=3
        )

        return "\n".join(r["content"] for r in response["results"])