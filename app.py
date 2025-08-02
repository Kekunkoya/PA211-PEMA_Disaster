import os
import json
import pickle
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------
# Utility: Ensure API Key Exists
# ----------------------
def ensure_api_key(service_name, key_name):
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"{service_name} API key is missing. Please enter it in the sidebar.")
    return api_key

# ----------------------
# Embedding Cache Utilities
# ----------------------
def cache_path_for_model(model_name):
    return f"{model_name.lower()}_cache.pkl"

def load_cache(model_name):
    path = cache_path_for_model(model_name)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def save_cache(model_name, embeddings, texts):
    path = cache_path_for_model(model_name)
    try:
        with open(path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "texts": texts}, f)
    except Exception as e:
        st.error(f"Failed to save cache for {model_name}: {e}")

# ----------------------
# Config
# ----------------------
DATA_FILE = "PA211_dataset.json"

# ----------------------
# Dataset Loader
# ----------------------
def load_dataset():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [f"{item['question']}\n{item['ideal_answer']}" for item in data]
    return data, texts

# ----------------------
# Embedding Builders
# ----------------------
def build_embeddings_openai(texts):
    ensure_api_key("OpenAI", "OPENAI_API_KEY")
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = []
    for txt in texts:
        try:
            truncated_txt = txt[:3000]
            resp = client.embeddings.create(model="text-embedding-3-small", input=truncated_txt)
            embeddings.append(resp.data[0].embedding)
        except Exception as e:
            st.error(f"OpenAI embedding error: {e}")
            raise RuntimeError(f"OpenAI embedding error: {e}")
    return np.array(embeddings)

def build_embeddings_gemini(texts):
    ensure_api_key("Gemini", "GEMINI_API_KEY")
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = "models/embedding-001"  # Stable Gemini embedding model
    embeddings = []
    for txt in texts:
        try:
            truncated_txt = txt[:3000]  # Prevent token overflow
            resp = genai.embed_content(model=model, content=truncated_txt)
            embeddings.append(resp["embedding"])
        except Exception as e:
            st.error(f"Gemini embedding error for: {txt[:50]}... — {e}")
            raise RuntimeError(f"Gemini embedding error: {e}")
    return np.array(embeddings)

# ----------------------
# Retrieval
# ----------------------
def retrieve(query, embeddings, texts, api_choice):
    if api_choice == "OpenAI":
        q_emb = build_embeddings_openai([query])[0].reshape(1, -1)
    else:
        q_emb = build_embeddings_gemini([query])[0].reshape(1, -1)
    sims = cosine_similarity(q_emb, embeddings)[0]
    ranked_idx = np.argsort(sims)[::-1]
    return ranked_idx, sims

# ----------------------
# Answer Generation
# ----------------------
def generate_answer_openai(query, context):
    ensure_api_key("OpenAI", "OPENAI_API_KEY")
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""Answer the question based on the context below.
Context:
{context}
Question: {query}
Answer:"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI generation error: {e}")
        raise RuntimeError(f"OpenAI generation error: {e}")

def generate_answer_gemini(query, context):
    ensure_api_key("Gemini", "GEMINI_API_KEY")
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Answer the question based on the context below.
Context:
{context}
Question: {query}
Answer:"""
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        st.error(f"Gemini generation error: {e}")
        raise RuntimeError(f"Gemini generation error: {e}")

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="PA 211 RAG Assistant", layout="centered")
st.title("PA 211 RAG Assistant")
st.markdown("Ask questions about PA 211 resources and get answers from OpenAI or Gemini with Retrieval-Augmented Generation.")

# Sidebar for API keys
st.sidebar.header("API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

# API choice
api_choice = st.radio("Choose API", ["OpenAI", "Gemini"])
top_k = st.slider("Number of retrieved documents (Top K)", min_value=1, max_value=10, value=3)

# Load dataset
data, texts = load_dataset()

# Load or Build cache
cache_data = load_cache(api_choice)
if cache_data and cache_data.get("texts") == texts:
    embeddings = cache_data["embeddings"]
    st.sidebar.success(f"Loaded cached embeddings for {api_choice}")
else:
    st.sidebar.info(f"Building embeddings for {api_choice}...")
    embeddings = build_embeddings_openai(texts) if api_choice == "OpenAI" else build_embeddings_gemini(texts)
    save_cache(api_choice, embeddings, texts)
    st.sidebar.success(f"Saved cache for {api_choice}")

# Query
query = st.text_area("Enter your query")
if st.button("Get Answer"):
    if embeddings is None:
        st.error("Please build the embedding cache first.")
    else:
        idxs, sims = retrieve(query, embeddings, texts, api_choice)
        top_contexts = [texts[i] for i in idxs[:top_k]]
        context_str = "\n\n".join(top_contexts)

        if api_choice == "OpenAI":
            answer = generate_answer_openai(query, context_str)
        else:
            answer = generate_answer_gemini(query, context_str)

        st.markdown("### Answer")
        st.write(answer)
