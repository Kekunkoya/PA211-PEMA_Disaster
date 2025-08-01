import os
import pickle
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_index(index_path, docstore_path):
    index = faiss.read_index(index_path)
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)
    return index, docstore

def retrieve_context(query, index, docstore, k=5):
    from openai import OpenAI
    embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding = embed_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    D, I = index.search([embedding], k)
    return [docstore[i] for i in I[0]]

def generate_answer_openai(query, index, docstore, retrieval_mode="simple", k=5):
    context_docs = retrieve_context(query, index, docstore, k)
    context = "\n\n".join(context_docs)
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
