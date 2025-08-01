import os
import pickle
import faiss
import streamlit as st
import google.generativeai as genai

@st.cache_resource
def load_index(index_path="faiss_index.idx", docstore_path="docstore.pkl"):
    index = faiss.read_index(index_path)
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)
    return index, docstore

def retrieve_context(query, index, docstore, top_k=3):
    # Gemini embeddings
    embedding = genai.embed_content(
        model="models/text-embedding-001",
        content=query
    )["embedding"]

    scores, ids = index.search([embedding], top_k)
    return [docstore[i] for i in ids[0]]

def gemini_rag(query):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    index, docstore = load_index()
    context_chunks = retrieve_context(query, index, docstore)
    context_text = "\n".join(context_chunks)

    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)
    return resp.text

