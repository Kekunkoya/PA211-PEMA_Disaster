import os
import pickle
import faiss
import streamlit as st
from openai import OpenAI

@st.cache_resource
def load_index(index_path="faiss_index.idx", docstore_path="docstore.pkl"):
    index = faiss.read_index(index_path)
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)
    return index, docstore

def retrieve_context(query, index, docstore, top_k=3):
    # Embed query using OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scores, ids = index.search([embedding], top_k)
    return [docstore[i] for i in ids[0]]

def openai_rag(query):
    index, docstore = load_index()
    context_chunks = retrieve_context(query, index, docstore)
    context_text = "\n".join(context_chunks)

    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content

