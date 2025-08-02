import os
import json
import pickle
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

# ----------------------
# Config
# ----------------------
DATA_FILE = "PA211_dataset.json"
OPENAI_CACHE_FILE = "openai_embeddings.pkl"
GEMINI_CACHE_FILE = "gemini_embeddings.pkl"

# ----------------------
# Load Dataset
# ----------------------
def load_dataset():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [f"{item['question']}\n{item['ideal_answer']}" for item in data]
    return data, texts

# ----------------------
# Load Cache
# ----------------------
def load_cache(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

# ----------------------
# Retrieval
# ----------------------
def retrieve(query_emb, embeddings, texts, top_k=3):
    sims = cosine_similarity(query_emb.reshape(1, -1), embeddings)[0]
    ranked_idx = np.argsort(sims)[::-1]
    return [(texts[i], sims[i]) for i in ranked_idx[:top_k]]

# ----------------------
# Build Query Embedding
# ----------------------
def build_query_embedding_openai(query):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model="text-embedding-3-small", input=query)
    return np.array(resp.data[0].embedding)

def build_query_embedding_gemini(query):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = "models/text-embedding-004"
    resp = genai.embed_content(model=model, content=query)
    return np.array(resp["embedding"])

# ----------------------
# Generate Answer
# ----------------------
def generate_openai(query, context):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"Answer based on the context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def generate_gemini(query, context):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Answer based on the context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    resp = model.generate_content(prompt)
    return resp.text.strip()

# ----------------------
# Evaluation Metrics
# ----------------------
def evaluate_responses(reference, candidate):
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)

    # Force BERTScore to run on CPU to avoid GPU errors on Streamlit Cloud
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False, device="cpu", model_type="microsoft/deberta-base-mnli")
    return bleu, float(P[0]), float(R[0]), float(F1[0])

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="RAG Evaluation - OpenAI vs Gemini", layout="wide")
st.title("RAG Evaluation: OpenAI vs Gemini")

st.sidebar.header("API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

top_k = st.sidebar.slider("Number of retrieved documents", 1, 5, 3)

# Load data
data, texts = load_dataset()
openai_cache = load_cache(OPENAI_CACHE_FILE)
gemini_cache = load_cache(GEMINI_CACHE_FILE)

if openai_cache is None or gemini_cache is None:
    st.error("Embedding caches missing! Please run the main RAG app first.")
    st.stop()

query = st.text_area("Enter your query")
if st.button("Run Evaluation"):
    if not query.strip():
        st.error("Please enter a query.")
        st.stop()

    # Build embeddings for query
    q_emb_openai = build_query_embedding_openai(query)
    q_emb_gemini = build_query_embedding_gemini(query)

    # Retrieve top docs
    top_docs_openai = retrieve(q_emb_openai, openai_cache, texts, top_k)
    top_docs_gemini = retrieve(q_emb_gemini, gemini_cache, texts, top_k)

    context_openai = "\n\n".join([doc for doc, _ in top_docs_openai])
    context_gemini = "\n\n".join([doc for doc, _ in top_docs_gemini])

    # Generate answers
    openai_answer = generate_openai(query, context_openai)
    gemini_answer = generate_gemini(query, context_gemini)

    # Reference answer
    reference = data[0]["ideal_answer"]

    # Evaluate
    openai_scores = evaluate_responses(reference, openai_answer)
    gemini_scores = evaluate_responses(reference, gemini_answer)

    # Display results side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("OpenAI Answer")
        st.write(openai_answer)
        st.markdown(f"**BLEU:** {openai_scores[0]:.4f} | **BERTScore F1:** {openai_scores[3]:.4f}")

    with col2:
        st.subheader("Gemini Answer")
        st.write(gemini_answer)
        st.markdown(f"**BLEU:** {gemini_scores[0]:.4f} | **BERTScore F1:** {gemini_scores[3]:.4f}")
