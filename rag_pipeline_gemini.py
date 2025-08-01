# rag_pipeline_gemini.py

import os
import pickle
import faiss
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Load Gemini FAISS Index ---
def load_index():
    index_path = "faiss_index_gemini.idx"
    store_path = "docstore_gemini.pkl"

    if not os.path.exists(index_path) or not os.path.exists(store_path):
        raise FileNotFoundError(f"Missing FAISS index or docstore for Gemini. Please run build_dual_faiss_indexes.py first.")

    index = faiss.read_index(index_path)
    with open(store_path, "rb") as f:
        docstore = pickle.load(f)

    return index, docstore

# --- Retrieve context from Gemini FAISS ---
def retrieve_context(query, top_k=3):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    index, docstore = load_index()
    vectorstore = FAISS(index=index, docstore=docstore, embeddings=embeddings)

    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

# --- Run Gemini RAG ---
def gemini_rag(query):
    context = retrieve_context(query)
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(
        f"Use this context to answer:\n\n{context}\n\nQuestion: {query}"
    )
    return response.text



