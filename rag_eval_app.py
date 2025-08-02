
import os
import json
import pickle
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from rouge import Rouge

# Load dataset
DATA_FILE = "PA211_dataset.json"
OPENAI_CACHE_FILE = "openai_embeddings.pkl"
GEMINI_CACHE_FILE = "gemini_embeddings.pkl"

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
texts = [f"{item['question']}\n{item['ideal_answer']}" for item in data]
references = [item["ideal_answer"] for item in data]

# Load cached embeddings
def load_cache(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

openai_embeddings = load_cache(OPENAI_CACHE_FILE)
gemini_embeddings = load_cache(GEMINI_CACHE_FILE)

# Ensure embeddings exist
if openai_embeddings is None or gemini_embeddings is None:
    st.error("Embedding cache not found. Please run the main RAG app first to build embeddings.")
    st.stop()

# API Setup
from openai import OpenAI
import google.generativeai as genai

openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

# Initialize APIs
openai_client = None
if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Helper: Build single query embedding
def build_query_embedding(query, api_choice):
    if api_choice == "OpenAI":
        resp = openai_client.embeddings.create(model="text-embedding-3-small", input=query)
        return np.array(resp.data[0].embedding)
    else:
        resp = genai.embed_content(model="models/text-embedding-004", content=query)
        return np.array(resp["embedding"])

# Retrieval
def retrieve(query, embeddings):
    q_emb = query.reshape(1, -1)
    sims = cosine_similarity(q_emb, embeddings)[0]
    ranked_idx = np.argsort(sims)[::-1]
    return ranked_idx, sims

# Generation
def generate_answer_openai(query, context):
    prompt = f"Answer the question based on the context below.\nContext:\n{context}\nQuestion: {query}\nAnswer:"
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content

def generate_answer_gemini(query, context):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Answer the question based on the context below.\nContext:\n{context}\nQuestion: {query}\nAnswer:"
    resp = model.generate_content(prompt)
    return resp.text

# Evaluation
def evaluate_metrics(reference, candidate):
    # BLEU
    bleu = sentence_bleu([reference.split()], candidate.split())
    # BERTScore
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
    # ROUGE-L
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference)[0]["rouge-l"]["f"]
    return {
        "BLEU": round(bleu, 3),
        "BERTScore_F1": round(F1.mean().item(), 3),
        "ROUGE-L_F": round(rouge_scores, 3)
    }

# Streamlit UI
st.set_page_config(page_title="RAG Evaluation", layout="wide")
st.title("RAG Evaluation App (OpenAI vs Gemini)")
st.markdown("This app compares OpenAI and Gemini RAG answers using BLEU, BERTScore, and ROUGE-L.")

query = st.text_area("Enter your query")

if st.button("Evaluate"):
    if not query.strip():
        st.error("Please enter a query.")
        st.stop()

    # Retrieve top-3 for each
    q_openai = build_query_embedding(query, "OpenAI")
    idxs_o, _ = retrieve(q_openai, openai_embeddings)
    context_o = "\n\n".join([texts[i] for i in idxs_o[:3]])

    q_gemini = build_query_embedding(query, "Gemini")
    idxs_g, _ = retrieve(q_gemini, gemini_embeddings)
    context_g = "\n\n".join([texts[i] for i in idxs_g[:3]])

    # Generate answers
    ans_openai = generate_answer_openai(query, context_o)
    ans_gemini = generate_answer_gemini(query, context_g)

    # Reference answer
    ref_answer = references[idxs_o[0]]

    # Evaluate
    eval_openai = evaluate_metrics(ref_answer, ans_openai)
    eval_gemini = evaluate_metrics(ref_answer, ans_gemini)

    # Display
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**OpenAI Answer:**")
        st.write(ans_openai)
        st.write("**Scores:**", eval_openai)
    with col2:
        st.markdown("**Gemini Answer:**")
        st.write(ans_gemini)
        st.write("**Scores:**", eval_gemini)
