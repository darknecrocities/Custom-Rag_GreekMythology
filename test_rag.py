import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from google import genai

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ----------------------------
# Load FAISS index and text chunks
# ----------------------------
index = faiss.read_index("greek_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# ----------------------------
# Gemini model (valid ID)
# ----------------------------
MODEL_NAME = "models/gemini-2.5-flash"  # best reasoning quality
EMBED_MODEL = "text-embedding-004"   # embeddings

# ----------------------------
# Function to test RAG
# ----------------------------
def ask_greek_bot(query, k=5):
    # 1️⃣ Embed query
    emb_response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query
    )
    q_emb = np.array([emb_response.embeddings[0].values], dtype="float32")

    # 2️⃣ FAISS search
    D, I = index.search(q_emb, k)
    
    print("\n--- Top-k Retrieved Chunks ---")
    for rank, idx in enumerate(I[0]):
        print(f"[{rank+1}] Distance: {D[0][rank]:.4f}")
        print(chunks[idx][:300], "...\n")  # print first 300 chars

    # 3️⃣ Prepare prompt with context
    context = "\n".join([chunks[i] for i in I[0]])
    prompt = f"""
You are an expert historian and mythologist specializing in Ancient Greek history and mythology.

Use ONLY the context below to answer accurately and with historical reasoning.

Context (top {k} chunks):
{context}

Question:
{query}

Answer clearly and cite the context if possible.
"""

    # 4️⃣ Generate answer
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text

# ----------------------------
# CLI loop for testing
# ----------------------------
if __name__ == "__main__":
    print("=== Greek History & Mythology RAG Tester ===")
    print("Type 'exit' to quit.")
    
    while True:
        q = input("\nYour question: ")
        if q.lower() in ("exit", "quit"):
            break
        answer = ask_greek_bot(q)
        print("\n--- AI Answer ---")
        print(answer)
