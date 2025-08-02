
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- Load dataset and build embeddings cache ---
dataset_path = os.path.join(os.path.dirname(__file__), "PA211_dataset.json")
doc_texts = []
doc_embeddings = None

def build_cache():
    global doc_texts, doc_embeddings
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    doc_texts = [f"Q: {d['question']}\nA: {d['ideal_answer']}" 
                 for d in data if 'question' in d and 'ideal_answer' in d]

    print(f"Building embedding cache for {len(doc_texts)} documents...")
    doc_embeddings = [genai.embed_content(model="models/text-embedding-004", content=text)["embedding"]
                      for text in doc_texts]
    doc_embeddings = np.array(doc_embeddings)
    print("Embedding cache built.")

# --- Retrieval ---
def retrieve_context(query, top_k=3):
    global doc_embeddings
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    if doc_embeddings is None:
        build_cache()
    query_emb = genai.embed_content(model="models/text-embedding-004", content=query)["embedding"]
    sims = cosine_similarity([query_emb], doc_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [doc_texts[i] for i in top_indices]

# --- Main entry point for Gemini RAG ---
def main(query: str, top_k: int = 3):
    context_docs = retrieve_context(query, top_k=top_k)
    context_str = "\n\n".join(context_docs)
    prompt = f"Use the following context to answer the query:\n{context_str}\n\nQuery: {query}"

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"


