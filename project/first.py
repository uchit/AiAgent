
import importlib.util
import sys

if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

chroma_client = chromadb.PersistentClient(path="chromadb")

# List existing collections
existing_collections = chroma_client.list_collections()  # Assuming this method exists

if "uchitCollection" in existing_collections:
    print("Collection 'uchitCollection' already exists.")
    # You can choose to use the existing collection or create a new one
else:
    # Create the collection if it doesn't exist
    collection = chroma_client.create_collection("uchitCollection7")
    print("Collection 'uchitCollection' created successfully!")


data_dir = '/workspace/Code/project/starter/games'

# Check if the directory exists, if not, create it
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Directory '{data_dir}' created.")

# Now you can list files in the specified directory
for file_name in sorted(os.listdir(data_dir)):
    print(file_name)


for file_name in sorted(os.listdir(data_dir)):
    if not file_name.endswith(".json"):
        continue

    file_path = os.path.join(data_dir, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        game = json.load(f)

    # You can change what text you want to index
    content = f"[{game['Platform']}] {game['Name']} ({game['YearOfRelease']}) - {game['Description']}"

    # Use file name (like 001) as ID
    doc_id = os.path.splitext(file_name)[0]

    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[game]
    )
