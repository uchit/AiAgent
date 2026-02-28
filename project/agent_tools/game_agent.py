import logging
from .agent_tools import retrieve_from_db, evaluate_results, TavilySearch

logging.basicConfig(level=logging.INFO)

class GameAgent:
    def __init__(self, vector_db, tavily_api_key: str = None):
        self.vector_db = vector_db
        self.tavily = TavilySearch(api_key=tavily_api_key) if tavily_api_key else None
        self.history = []

    def answer_query(self, query: str, n_results: int = 3):
        logging.info("Using internal DB retrieval...")
        db_results = retrieve_from_db(self.vector_db, query, n_results=n_results)

        needs_web = evaluate_results(db_results)
        answer = {}

        if not needs_web:
            # Format DB results
            answer["source"] = "internal_db"
            docs = db_results["documents"][0]
            metas = db_results["metadatas"][0]
            answer["results"] = [{"doc": d, "meta": m} for d, m in zip(docs, metas)]
        else:
            # Fallback to Tavily
            if self.tavily:
                logging.info("Falling back to Tavily web search...")
                web_results = self.tavily.search(query, n_results=n_results)
                answer["source"] = "web_search"
                answer["results"] = web_results
            else:
                answer["source"] = "none"
                answer["results"] = []

        # Save to conversation history
        self.history.append({"query": query, "answer": answer})
        return answer