import os
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from google import genai

# Load environment
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 1️⃣ Read all PDFs
def read_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# Collect documents
docs = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        docs.append(read_pdf(os.path.join("data", file)))

# 2️⃣ Chunk text
def chunk_text(text, size=500):
    return [text[i:i + size] for i in range(0, len(text), size)]

all_chunks = []
for d in docs:
    all_chunks.extend(chunk_text(d))

print(f"Total chunks: {len(all_chunks)}")

# 3️⃣ Create embeddings using NEW Gemini API
embeddings = []

for chunk in all_chunks:
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=chunk
    )
    embeddings.append(response.embeddings[0].values)

# 4️⃣ Create FAISS index
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
vectors = np.array(embeddings).astype("float32")
index.add(vectors)

# Save index & chunks
faiss.write_index(index, "greek_index.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ Gemini RAG index built successfully!")
