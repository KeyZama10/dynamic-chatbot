import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to your existing database
client = chromadb.PersistentClient(path="vectordb")
collection = client.get_or_create_collection("knowledge_base")

# 3. Setup the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def update(folder):
    if not os.path.exists(folder):
        print(f"Error: The folder '{folder}' was not found.")
        return

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        # Skip folders if any exist inside 'updates'
        if os.path.isdir(file_path):
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = text_splitter.split_text(text)
        embeddings = model.encode(chunks)

        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                embeddings=[embeddings[i].tolist()],
                ids=[f"update_{file}_{i}"],
                metadatas=[{"source": file}]
            )

# 4. Run the update on the 'updates' folder
update("data/updates")
print("Knowledge base updated with new information.")