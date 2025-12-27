__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from sentence_transformers import SentenceTransformer
import os

# 1. Load the AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to the database with a specific cloud path
persist_path = "./vectordb"
client = chromadb.PersistentClient(path=persist_path)

# 3. Fail-safe collection retrieval
collection = client.get_or_create_collection("knowledge_base")

def ask_question(query):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    if not results["documents"] or not results["documents"][0]:
        return "The AI brain is currently empty. You need to upload your data files to GitHub."

    context = "\n".join(results["documents"][0])
    return f"\nBased on the available knowledge:\n{context}"