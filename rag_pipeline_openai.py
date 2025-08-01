import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import pickle

load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Load FAISS index
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH_OPENAI", "faiss_index_openai")
index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "index.faiss"))

# Load document mapping
with open(os.path.join(FAISS_INDEX_PATH, "docs.pkl"), "rb") as f:
    documents = pickle.load(f)

def retrieve_context(query, k=5):
    """Retrieve top-k relevant docs for a query."""
    # Embed query
    embed = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    D, I = index.search([embed], k)
    results = [documents[i] for i in I[0]]
    return results

def generate_answer(query, k=5):
    """Generate answer using OpenAI GPT with retrieved context."""
    context_docs = retrieve_context(query, k)
    context_text = "\n\n".join(context_docs)

    prompt = f"Answer the following question based on the context:\n\nContext:\n{context_text}\n\nQuestion:\n{query}"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant using RAG."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content
