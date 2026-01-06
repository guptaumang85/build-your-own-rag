# Build Your Own RAG

A hands-on Retrieval-Augmented Generation (RAG) system built as part of the **Retrieval-Augmented Generation**.

This repository walks through building a RAG pipeline that:
- Retrieves relevant text from a dataset
- Generates responses using a large language model
- Combines retrieval with generation to answer queries

---

## 🧠 Overview

RAG (Retrieval-Augmented Generation) combines:
1. **Dense Retrieval** — finding relevant context for a query
2. **Generative Models** — producing fluent responses using context

This implementation uses:
- Embeddings for retrieval
- An LLM (Together.ai, OpenAI, or HF models) for generation
- Vector store for efficient search

---

## 🚀 Features

✔ Compute embeddings for a dataset  
✔ Vector similarity search  
✔ Query-based retrieval  
✔ LLM generation with context  
✔ Integration with Together.ai / OpenAI API  

---

## 🛠 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/guptaumang85/build-your-own-rag.git
cd build-your-own-rag
python -m venv rag-venv
source rag-venv/bin/activate      # macOS/Linux
# rag-venv\Scripts\activate        # Windows
pip install --upgrade pip
pip install -r requirements.txt

Create a .env file (NOT committed):
# API Keys
TOGETHER_API_KEY=
```
You need to have these keys in your .env file to call an llm. You need to create
account and purchase. Min $5.
