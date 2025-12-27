import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to the database in your 'vectordb' folder
client = chromadb.PersistentClient(path="vectordb")
collection = client.get_or_create_collection("knowledge_base")

def ask_question(query):
    # Convert your question into a math format the AI understands
    query_embedding = model.encode([query]).tolist()

    # Look for the top 3 most relevant pieces of information
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    # Combine the found text into one answer
    context = "\n".join(results["documents"][0])

    if not context.strip():
        return "I don't have enough information to answer this yet."

    return f"\nBased on the available knowledge:\n{context}"

# 3. Create a loop so you can keep asking questions
print("--- AI Chatbot Started! (Type 'quit' to stop) ---")
while True:
    user_query = input("\nAsk a question: ")
    if user_query.lower() == 'quit':
        break
    
    answer = ask_question(user_query)
    print(answer)