# 🏛️ Greek History & Mythology RAG Chatbot

## 🌟 Project Overview

This project is a **Retrieval-Augmented Generation (RAG) chatbot** specialized in **Ancient Greek history and mythology**.  

It uses:

- ⚡ **FAISS** for similarity search over PDF chunks.  
- 🧠 **Embeddings** via `text-embedding-004` for both documents and queries.  
- 🤖 **Google Gemini LLM** to generate accurate answers grounded in the retrieved context.  

The chatbot ensures that responses are **factual and context-aware**, based on your PDF datasets.

---

## ✨ Features

- 🏺 Ask questions about Greek history, mythology, gods, heroes, and events.  
- 🚀 FAISS-based fast similarity search over large text datasets.  
- 💡 Gemini LLM provides reasoning-based, detailed answers.  
- 📚 Easy to expand with new PDF documents.

---

## 🛠️ Project Workflow

1. **📂 Data Preparation**  
   - Collect PDFs (Greek history and mythology) in the `data/` folder.

2. **📄 Text Extraction & Chunking**  
   - Extract text using `PyPDF2`.  
   - Chunk text (~500 characters) to optimize embeddings.

3. **🔗 Embedding & Indexing**  
   - Convert chunks to embeddings via Gemini `text-embedding-004`.  
   - Build FAISS index and save (`greek_index.faiss`).  
   - Pickle the text chunks (`chunks.pkl`).

4. **🤖 Query & Answer**  
   - Embed user query.  
   - Retrieve top-k chunks from FAISS.  
   - Generate answer via Gemini LLM using only retrieved context.

5. **🧪 Testing**  
   - CLI-based testing prints retrieved chunks, distances, and the generated answer.


---

## 📂 Project Structure
```
greekmodel/
│
├─ data/                 # 📄 PDFs for Greek history/mythology
│   ├─ greek_history.pdf
│   └─ greek_myths.pdf
│
├─ build_rag_index.py    # 🛠️ Build embeddings and FAISS index
├─ chatbot.py            # 💬 Main RAG chatbot for CLI
├─ test_rag.py           # 🧪 Test and debug RAG pipeline
├─ chunks.pkl            # 💾 Pickled text chunks
├─ greek_index.faiss     # 💾 FAISS vector index
├─ .env                  # 🔑 Gemini API key
└─ README.txt            # 📖 This documentation
```

---

## 🔧 Functions Overview

### build_rag_index.py

- `read_pdf(file_path)` – Extract text from a PDF file.
- `chunk_text(text, size=500)` – Split text into smaller chunks.
- Workflow: Load PDFs → Chunk text → Generate embeddings → Build FAISS index → Save index and chunks.

### chatbot.py

- `ask_greek_bot(query, k=5)` – Embed query, retrieve top-k chunks, generate answer using Gemini LLM.
- CLI loop to input questions and print answers.

### test_rag.py

- Test the RAG pipeline: shows top-k retrieved chunks with distances and generates the LLM answer.

---

## ✅ Requirements

- Python 3.12+  
- Packages:
- faiss-cpu
- numpy
- PyPDF2
- python-dotenv
- google-genai

## ⚡ Setup Instructions

### 1️⃣ Clone project  
```
git clone <your-repo-url>
cd greekmodel
```
### 2️⃣ Create virtual environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Add Gemini API key
Create a .env file in the project root:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```
### 5️⃣ Add PDFs
Place Greek history/mythology PDFs inside the data/ folder.

### 6️⃣ Build FAISS index
```
python build_rag_index.py
```
### 7️⃣ Run chatbot
```
python chatbot.py
```
### 8️⃣ Test RAG pipeline
```
python test_rag.py
```
### 💬 Example Questions
```

“Who was the king of the Greek gods?”

“What happened during the Trojan War?”

“Describe the Spartan army.”
```

### ⚠️ Notes
```
- The AI answers only using context retrieved from your PDFs.
- Add more PDFs and rebuild the index to expand knowledge.
```
