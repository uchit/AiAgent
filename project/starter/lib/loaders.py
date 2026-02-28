import json

def load_games(path: str):
    with open(path, "r") as f:
        return json.load(f)