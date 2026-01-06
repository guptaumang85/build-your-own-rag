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

You'll build a RAG pipeline using a dataset containing news information from BBC News. The goal is to enable the LLM to retrieve relevant news details from the dataset and use that information to generate more informed responses. The model you'll be using is the llama-3-1-8b-instruct-turbo, which was trained on data up to December 2023. The idea is to create a RAG system that allows it to include information on events that occurred in 2024.

---

## 📊 Dataset

This project uses the **News Headlines 2024** dataset from Kaggle.

**Dataset link:**  
https://www.kaggle.com/datasets/dylanjcastillo/news-headlines-2024

### 📌 About the Dataset
The dataset contains **thousands of news headlines from BBC News**, along with related metadata. It is well-suited for building and evaluating Retrieval-Augmented Generation (RAG) systems because it provides:

- Short, information-dense text (headlines)
- Real-world news content
- Sufficient volume for embedding, indexing, and retrieval

### 🧾 Key Columns (Typical)
- `headline` – The news headline text  
- `description` / `short_description` – Additional context (if available)  
- `date` – Publication date  
- `source` – News source (BBC News)

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

Create a .env file (NOT committed):
# API Keys
TOGETHER_API_KEY=
```
You need to have these key in your .env file to call an llm. You need to create
account and purchase. Min $5.

Install required packages and
Run **rag_implementation.py**
