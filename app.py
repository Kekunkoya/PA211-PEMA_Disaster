import os
import time
import logging
from dotenv import load_dotenv
import faiss
import numpy as np
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index_openai")
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.75))

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)

# -----------------------
# 1. Load FAISS index
# -----------------------
def load_index():
    logging.info(f"Loading FAISS index from {FAISS_INDEX_PATH} ...")
    index = faiss.read_index(f"{FAISS_INDEX_PATH}/index.faiss")
    with open(f"{FAISS_INDEX_PATH}/docs.txt", "r", encoding="utf-8") as f:
        documents = f.readlines()
    return index, documents

# -----------------------
# 2. Query Transformation
# -----------------------
def transform_query(query):
    try:
        logging.info("Transforming query ...")
        prompt = f"Rewrite this user query to be more precise for document retrieval:\n\n{query}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a query rewriting assistant."},
                      {"role": "user", "content": prompt}]
        )
        new_query = resp.choices[0].message.content.strip()
        logging.info(f"Transformed query: {new_query}")
        return new_query
    except Exception as e:
        logging.warning(f"Query transformation failed: {e}")
        return query

# -----------------------
# 3. HyDE synthetic query
# -----------------------
def hyde_query(query):
    try:
        logging.info("Generating HyDE synthetic query ...")
        prompt = f"Generate a hypothetical answer to this query to help retrieve documents:\n\n{query}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        synthetic_query = resp.choices[0].message.content.strip()
        return synthetic_query
    except Exception as e:
        logging.warning(f"HyDE generation failed: {e}")
        return query

# -----------------------
# 4. Fusion Retrieval
# -----------------------
def retrieve_context(index, documents, query, top_k=5):
    logging.info("Retrieving documents ...")
    # Embed query
    emb = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    emb = np.array(emb, dtype=np.float32).reshape(1, -1)
    scores, idxs = index.search(emb, top_k)

    results = [(documents[i], float(scores[0][pos])) for pos, i in enumerate(idxs[0])]
    for doc, score in results:
        logging.info(f"Score: {score:.4f} | Doc: {doc[:80]}...")
    return results

# -----------------------
# 5. Context Enrichment
# -----------------------
def enrich_context(results):
    context_texts = [doc for doc, _ in results]
    return "\n\n".join(context_texts)

# -----------------------
# 6. Generate Answer
# -----------------------
def generate_answer(query):
    start_time = time.time()

    index, documents = load_index()
    transformed = transform_query(query)
    results = retrieve_context(index, documents, transformed)

    best_score = results[0][1] if results else 0
    if best_score < RETRIEVAL_SCORE_THRESHOLD:
        logging.info(f"Score {best_score:.4f} < {RETRIEVAL_SCORE_THRESHOLD} → Using HyDE")
        synthetic = hyde_query(query)
        results = retrieve_context(index, documents, synthetic)

    context = enrich_context(results)

    prompt = f"Answer the following query based on the context below.\n\nQuery: {query}\n\nContext:\n{context}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful disaster response assistant."},
                  {"role": "user", "content": prompt}]
    )

    answer = resp.choices[0].message.content.strip()
    logging.info(f"Total time: {time.time() - start_time:.2f}s")
    return answer
