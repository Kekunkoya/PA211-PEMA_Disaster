import os
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
import pickle

load_dotenv()

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("⚠️ Missing GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)

# Load FAISS index
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH_GEMINI", "faiss_index_gemini")
index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "index.faiss"))

# Load document mapping
with open(os.path.join(FAISS_INDEX_PATH, "docs.pkl"), "rb") as f:
    documents = pickle.load(f)

def retrieve_context(query, k=5):
    """Retrieve top-k relevant docs for a query."""
    embed = genai.embed_content(
        model="models/text-embedding-001",
        content=query
    )["embedding"]

    D, I = index.search([embed], k)
    results = [documents[i] for i in I[0]]
    return results

def generate_answer(query, k=5):
    """Generate answer using Gemini with retrieved context."""
    context_docs = retrieve_context(query, k)
    context_text = "\n\n".join(context_docs)

    prompt = f"Answer the following question based on the context:\n\nContext:\n{context_text}\n\nQuestion:\n{query}"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
