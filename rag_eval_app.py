# Let's rewrite rag_eval_app.py with the nltk punkt download at the top and ensure it's self-contained.

rag_eval_app_code = """
import os
import json
import pickle
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt')

from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# Utility functions
def load_cache(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

def build_embeddings_openai(texts):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = []
    for txt in texts:
        resp = client.embeddings.create(model="text-embedding-3-small", input=txt)
        embeddings.append(resp.data[0].embedding)
    return np.array(embeddings)

def build_embeddings_gemini(texts):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = "models/text-embedding-004"
    embeddings = []
    for txt in texts:
        resp = genai.embed_content(model=model, content=txt)
        embeddings.append(resp["embedding"])
    return np.array(embeddings)

def retrieve(query, embeddings, texts, api_choice):
    q_emb = build_embeddings_openai([query])[0].reshape(1, -1) if api_choice == "OpenAI" else build_embeddings_gemini([query])[0].reshape(1, -1)
    sims = cosine_similarity(q_emb, embeddings)[0]
    ranked_idx = np.argsort(sims)[::-1]
    return ranked_idx, sims

def generate_answer_openai(query, context):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f\"\"\"Answer the question based on the context below.
Context:
{context}
Question: {query}
Answer:\"\"\"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content

def generate_answer_gemini(query, context):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f\"\"\"Answer the question based on the context below.
Context:
{context}
Question: {query}
Answer:\"\"\"
    resp = model.generate_content(prompt)
    return resp.text

# Evaluation metrics
def evaluate_metrics(reference, candidate):
    bleu = sentence_bleu([reference.split()], candidate.split())
    P, R, F1 = bert_score([candidate], [reference], lang="en", rescale_with_baseline=True)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(reference, candidate)
    return bleu, float(P[0]), float(R[0]), float(F1[0]), rouge['rougeL'].fmeasure

# Streamlit UI
st.title("RAG Evaluation App")
st.sidebar.header("API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

DATA_FILE = "PA211_dataset.json"
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
texts = [f"{item['question']}\n{item['ideal_answer']}" for item in data]

# Build or load embeddings
openai_embeddings = load_cache("openai_embeddings.pkl") or build_embeddings_openai(texts)
gemini_embeddings = load_cache("gemini_embeddings.pkl") or build_embeddings_gemini(texts)

query = st.text_area("Enter a query for evaluation")
top_k = st.slider("Number of retrieved docs", 1, 5, 3)

if st.button("Evaluate"):
    idxs_oai, _ = retrieve(query, openai_embeddings, texts, "OpenAI")
    idxs_gem, _ = retrieve(query, gemini_embeddings, texts, "Gemini")

    ctx_oai = "\\n\\n".join([texts[i] for i in idxs_oai[:top_k]])
    ctx_gem = "\\n\\n".join([texts[i] for i in idxs_gem[:top_k]])

    ans_oai = generate_answer_openai(query, ctx_oai)
    ans_gem = generate_answer_gemini(query, ctx_gem)

    reference = data[idxs_oai[0]]["ideal_answer"]

    bleu_o, P_o, R_o, F1_o, rouge_o = evaluate_metrics(reference, ans_oai)
    bleu_g, P_g, R_g, F1_g, rouge_g = evaluate_metrics(reference, ans_gem)

    st.subheader("OpenAI Answer")
    st.write(ans_oai)
    st.write(f"BLEU: {bleu_o:.4f}, BERT-P: {P_o:.4f}, BERT-R: {R_o:.4f}, BERT-F1: {F1_o:.4f}, ROUGE-L: {rouge_o:.4f}")

    st.subheader("Gemini Answer")
    st.write(ans_gem)
    st.write(f"BLEU: {bleu_g:.4f}, BERT-P: {P_g:.4f}, BERT-R: {R_g:.4f}, BERT-F1: {F1_g:.4f}, ROUGE-L: {rouge_g:.4f}")
"""

# Save this file for the user
with open("/mnt/data/rag_eval_app_fixed.py", "w") as f:
    f.write(rag_eval_app_code)

"/mnt/data/rag_eval_app_fixed.py"
