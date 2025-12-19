import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Load FAISS index and chunks
index = faiss.read_index("greek_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# ✅ VALID Gemini model ID
MODEL_NAME = "models/gemini-2.5-flash"
# Or use this for higher reasoning quality:
# MODEL_NAME = "models/gemini-1.5-pro"


def ask_greek_bot(query, k=5):
    # Embed the query
    emb_response = client.models.embed_content(
        model="text-embedding-004",
        contents=query
    )

    q_emb = np.array(
        [emb_response.embeddings[0].values],
        dtype="float32"
    )

    # FAISS similarity search
    D, I = index.search(q_emb, k)
    context = "\n".join(chunks[i] for i in I[0])

    prompt = f"""
You are an expert historian and mythologist specializing in Ancient Greek history and mythology.

Use ONLY the context below to answer accurately and with clear reasoning.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# CLI loop
if __name__ == "__main__":
    while True:
        q = input("\nAsk about Greek history or mythology (type 'exit' to quit): ")
        if q.lower() in ("exit", "quit"):
            break
        print("\n", ask_greek_bot(q))
