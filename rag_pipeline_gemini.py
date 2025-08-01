import os
import pickle
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def load_index(index_path, docstore_path):
    index = faiss.read_index(index_path)
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)
    return index, docstore

def retrieve_context(query, index, docstore, k=5):
    embed_model = "models/text-embedding-001"
    embedding = genai.embed_content(
        model=embed_model,
        content=query
    )["embedding"]
    D, I = index.search([embedding], k)
    return [docstore[i] for i in I[0]]

def generate_answer_gemini(query, index, docstore, retrieval_mode="simple", k=5):
    context_docs = retrieve_context(query, index, docstore, k)
    context = "\n\n".join(context_docs)
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()

