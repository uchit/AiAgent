import os
import json
import logging
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

try:
    from .lib.vector_db import VectorDB
    from .lib.agent import GameAgent
except ImportError:
    from lib.vector_db import VectorDB
    from lib.agent import GameAgent

logging.basicConfig(level=logging.INFO)

PROJECT_DIR = Path(__file__).resolve().parent
GAMES_DIR = PROJECT_DIR.parent / "games"

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
    # Initialize DB
    db = VectorDB(
        collection_name="games_collection",
        persist_dir=str(PROJECT_DIR / "chromadb"),
    )

    # Load and ingest games
    games = load_games()
    db.ingest(games)

    # Initialize agent
    agent = GameAgent(vector_db=db)

    # Example queries
    queries = [
        "Open world superhero action game",
        "First Pokémon game released on Game Boy",
        "Classic platformer by Nintendo"
    ]

    for q in queries:
        logging.info(f"\nQuery: {q}")
        answer = agent.answer_query(q)
        logging.info(f"Agent Answer: {answer}\n{'-'*50}")

if __name__ == "__main__":
    main()
