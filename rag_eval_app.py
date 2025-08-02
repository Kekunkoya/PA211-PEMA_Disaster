import os
import json
import streamlit as st
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Download punkt for tokenization
nltk.download("punkt", quiet=True)

# ----------------------
# Load Dataset
# ----------------------
DATA_FILE = "PA211_dataset.json"

@st.cache_data
def load_dataset():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------
# API Key Handling
# ----------------------
def ensure_api_key(service_name, key_name):
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"{service_name} API key is missing. Please enter it in the sidebar.")
    return api_key

# ----------------------
# Model Calls
# ----------------------
def get_openai_answer(query, context):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"Answer the question based on the context below.\nContext:\n{context}\nQuestion: {query}\nAnswer:"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content

def get_gemini_answer(query, context):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Answer the question based on the context below.\nContext:\n{context}\nQuestion: {query}\nAnswer:"
    resp = model.generate_content(prompt)
    return resp.text

# ----------------------
# Evaluation Metrics
# ----------------------
def evaluate_responses(reference, candidate):
    # BLEU
    bleu = sentence_bleu([reference.split()], candidate.split())

    # ROUGE
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(reference, candidate)

    # BERTScore
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
    
    return {
        "BLEU": round(bleu, 4),
        "ROUGE-1": round(rouge_scores['rouge1'].fmeasure, 4),
        "ROUGE-L": round(rouge_scores['rougeL'].fmeasure, 4),
        "BERTScore_F1": round(F1[0].item(), 4)
    }

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="RAG Evaluation", layout="wide")
st.title("📊 RAG Model Evaluation - OpenAI vs Gemini")

# Sidebar API Keys
st.sidebar.header("API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

data = load_dataset()

query = st.text_input("Enter your query:")
top_k = st.slider("Number of context docs", 1, 5, 3)

if st.button("Run Evaluation"):
    if not query:
        st.error("Please enter a query.")
    else:
        context = "\n\n".join([f"{item['question']} {item['ideal_answer']}" for item in data[:top_k]])
        reference = data[0]["ideal_answer"]

        with st.spinner("Getting OpenAI answer..."):
            openai_answer = get_openai_answer(query, context)

        with st.spinner("Getting Gemini answer..."):
            gemini_answer = get_gemini_answer(query, context)

        openai_scores = evaluate_responses(reference, openai_answer)
        gemini_scores = evaluate_responses(reference, gemini_answer)

        st.subheader("OpenAI Answer")
        st.write(openai_answer)
        st.json(openai_scores)

        st.subheader("Gemini Answer")
        st.write(gemini_answer)
        st.json(gemini_scores)

        # Comparison Table
        st.subheader("📈 Comparison Table")
        st.table({
            "Metric": list(openai_scores.keys()),
            "OpenAI": list(openai_scores.values()),
            "Gemini": list(gemini_scores.values())
        })
