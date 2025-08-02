
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --- Load dataset and build embeddings cache ---
dataset_path = os.path.join(os.path.dirname(__file__), "PA211_dataset.json")
doc_texts = []
doc_embeddings = None

def build_cache():
    global doc_texts, doc_embeddings
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    doc_texts = [f"Q: {d['question']}\nA: {d['ideal_answer']}" 
                 for d in data if 'question' in d and 'ideal_answer' in d]

    print(f"Building embedding cache for {len(doc_texts)} documents...")
    doc_embeddings = [client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding 
                      for text in doc_texts]
    doc_embeddings = np.array(doc_embeddings)
    print("Embedding cache built.")

# --- Retrieval ---
def retrieve_context(query, top_k=3):
    global doc_embeddings
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if doc_embeddings is None:
        build_cache()
    query_emb = client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
    sims = cosine_similarity([query_emb], doc_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [doc_texts[i] for i in top_indices]

# --- Main entry point for OpenAI RAG ---
def main(query: str, top_k: int = 3):
    context_docs = retrieve_context(query, top_k=top_k)
    context_str = "\n\n".join(context_docs)
    prompt = f"Use the following context to answer the query:\n{context_str}\n\nQuery: {query}"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful PA 211 RAG assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI Error: {e}"
