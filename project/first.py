import os
import json
import logging
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

try:
    from .vector_db import VectorDB
except ImportError:
    from vector_db import VectorDB

from agent_tools.game_tools import VectorDBTool, ResultEvaluator, WebSearchTool
from agent_tools.game_agent import GameAgent

logging.basicConfig(level=logging.INFO)

PROJECT_DIR = Path(__file__).resolve().parent
GAMES_DIR = PROJECT_DIR / "games"

def load_games():
    games = []
    for file_name in sorted(os.listdir(GAMES_DIR)):
        if file_name.endswith(".json"):
            path = os.path.join(GAMES_DIR, file_name)
            with open(path, "r", encoding="utf-8") as f:
                g = json.load(f)
                if "Name" in g and "Description" in g:
                    games.append(g)
    logging.info(f"Found {len(games)} game entries.")
    return games

def main():
    # Initialize VectorDB
    db = VectorDB(collection_name="games_collection", persist_dir=str(PROJECT_DIR / "chromadb"))

    # Load and ingest games
    games = load_games()
    db.ingest(games)

    # Initialize tools
    vector_tool = VectorDBTool(db)
    evaluator = ResultEvaluator(distance_threshold=0.35)
    web_tool = WebSearchTool(api_key=os.environ.get("TAVILY_API_KEY"))

    # Initialize agent
    agent = GameAgent(vector_tool, evaluator, web_tool)

    # Example queries
    queries = [
        "When was Super Mario World released?",
        "Tell me more about the first game you mentioned.",
        "Which platform is Minecraft available on?",
    ]

    for q in queries:
        result = agent.ask(q)
        logging.info(f"\nQuery: {q}")
        logging.info(f"Answer: {result['answer']}")
        logging.info(f"Source: {result['source']}")
        logging.info(f"Reasoning: {result['reasoning']}")
        if result.get("effective_query") and result["effective_query"] != q:
            logging.info(f"Effective Query: {result['effective_query']}")
        if result.get("citations"):
            logging.info("Citations:")
            for c in result["citations"]:
                logging.info(f"  [{c['id']}] {c['label']} -> {c['source']}")

if __name__ == "__main__":
    main()
