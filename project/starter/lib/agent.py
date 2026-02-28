import logging
from .web_tools import TavilySearch, evaluate_results
from .vector_db import VectorDB

logging.basicConfig(level=logging.INFO)

def retrieve_from_db(vector_db: VectorDB, query: str, n_results: int = 3):
    return vector_db.search(query, n_results=n_results)

class GameAgent:
    """Stateful agent managing queries, internal DB, and web fallback."""
    def __init__(self, vector_db: VectorDB, tavily_api_key: str = None):
        self.vector_db = vector_db
        self.tavily = TavilySearch(api_key=tavily_api_key) if tavily_api_key else None
        self.history = []

    def answer_query(self, query: str, n_results: int = 3):
        logging.info("Using internal DB retrieval...")
        db_results = retrieve_from_db(self.vector_db, query, n_results=n_results)

        needs_web = evaluate_results(db_results)
        answer = {}

        if not needs_web:
            answer["source"] = "internal_db"
            docs = db_results["documents"][0]
            metas = db_results["metadatas"][0]
            answer["results"] = [{"doc": d, "meta": m} for d, m in zip(docs, metas)]
        else:
            if self.tavily:
                logging.info("Falling back to Tavily web search...")
                web_results = self.tavily.search(query, n_results=n_results)
                answer["source"] = "web_search"
                answer["results"] = web_results
            else:
                answer["source"] = "none"
                answer["results"] = []

        self.history.append({"query": query, "answer": answer})
        return answer