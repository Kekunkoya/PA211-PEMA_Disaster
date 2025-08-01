# rag_pipeline_openai.py

import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# --- Load OpenAI FAISS Index ---
def load_index():
    index_path = "faiss_index_openai.idx"
    store_path = "docstore_openai.pkl"

    if not os.path.exists(index_path) or not os.path.exists(store_path):
        raise FileNotFoundError(f"Missing FAISS index or docstore for OpenAI. Please run build_dual_faiss_indexes.py first.")

    index = faiss.read_index(index_path)
    with open(store_path, "rb") as f:
        docstore = pickle.load(f)

    return index, docstore

# --- Retrieve context from OpenAI FAISS ---
def retrieve_context(query, top_k=3):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    index, docstore = load_index()
    vectorstore = FAISS(index=index, docstore=docstore, embeddings=embeddings)

    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

# --- Run OpenAI RAG ---
def openai_rag(query):
    context = retrieve_context(query)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant using PA 211 disaster resources."},
            {"role": "user", "content": f"Use this context to answer:\n\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content



