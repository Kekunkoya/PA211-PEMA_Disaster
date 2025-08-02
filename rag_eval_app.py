# rag_notebook_comparison_app.py

import streamlit as st
import os
import json
import nbformat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ---------------- #
PA211_FILE = "PA211_dataset.json"
NOTEBOOK_OPENAI = "04_context_enriched_ragKemi.ipynb"
NOTEBOOK_GEMINI = "G04_context_enriched_ragKemi.ipynb"

# ---------------- FUNCTIONS ---------------- #
def extract_rag_answers(nb_path, questions):
    """Extract only RAG answers matching the dataset questions."""
    nb = nbformat.read(nb_path, as_version=4)
    answers = []
    
    for q in questions:
        found_answer = None
        for cell in nb.cells:
            if cell.cell_type == "code" and "outputs" in cell:
                for output in cell.outputs:
                    text_out = None
                    if output.output_type == "stream" and hasattr(output, "text"):
                        text_out = output.text.strip()
                    elif output.output_type == "execute_result" and "text/plain" in output.data:
                        text_out = output.data["text/plain"].strip()
                    
                    # Only keep non-trivial answers
                    if text_out and len(text_out.split()) > 3:
                        found_answer = text_out
                        break
            if found_answer:
                break
        answers.append(found_answer if found_answer else "")
    
    return answers

def cosine_score(text1, text2, embedder):
    emb = embedder.encode([text1, text2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

# ---------------- STREAMLIT UI ---------------- #
st.title("📊 RAG Model Output Comparison – OpenAI vs Gemini")

# Load dataset
try:
    with open(PA211_FILE, "r") as f:
        pa211_data = json.load(f)
except FileNotFoundError:
    st.error(f"❌ Could not find {PA211_FILE}. Please place it in the same folder.")
    st.stop()

questions = [item["question"] for item in pa211_data]

# Load aligned answers from notebooks
try:
    openai_outputs = extract_rag_answers(NOTEBOOK_OPENAI, questions)
    gemini_outputs = extract_rag_answers(NOTEBOOK_GEMINI, questions)
except FileNotFoundError as e:
    st.error(f"❌ Missing notebook file: {e}")
    st.stop()

# Alignment check
min_len = min(len(pa211_data), len(openai_outputs), len(gemini_outputs))
if len(openai_outputs) != len(gemini_outputs) or len(openai_outputs) != len(pa211_data):
    st.warning(f"⚠️ Length mismatch detected – trimming to {min_len} entries.")

df = pd.DataFrame({
    "Question": [item["question"] for item in pa211_data[:min_len]],
    "Reference Answer": [item["ideal_answer"] for item in pa211_data[:min_len]],
    "OpenAI Answer": openai_outputs[:min_len],
    "Gemini Answer": gemini_outputs[:min_len]
})

# Load embedding model
with st.spinner("Loading embedding model..."):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Compute similarity and error scores
df["OpenAI vs Ref"] = df.apply(lambda row: cosine_score(row["OpenAI Answer"], row["Reference Answer"], embedder), axis=1)
df["Gemini vs Ref"] = df.apply(lambda row: cosine_score(row["Gemini Answer"], row["Reference Answer"], embedder), axis=1)
df["OpenAI vs Gemini"] = df.apply(lambda row: cosine_score(row["OpenAI Answer"], row["Gemini Answer"], embedder), axis=1)

# Error = 1 - similarity
df["OpenAI Error"] = 1 - df["OpenAI vs Ref"]
df["Gemini Error"] = 1 - df["Gemini vs Ref"]
df["Cross Error"] = 1 - df["OpenAI vs Gemini"]

# Show table
st.subheader("🔍 Comparison Table")
st.dataframe(df)

# ---------------- HEATMAPS ---------------- #
st.subheader("🔥 Cosine Similarity Heatmap")
heatmap_data = df[["OpenAI vs Ref", "Gemini vs Ref", "OpenAI vs Gemini"]].T
plt.figure(figsize=(12, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Cosine Similarity Heatmap")
plt.xlabel("Question Index")
plt.ylabel("Comparison Type")
st.pyplot(plt)

st.subheader("🚨 Error Heatmap (1 - Similarity)")
error_data = df[["OpenAI Error", "Gemini Error", "Cross Error"]].T
plt.figure(figsize=(12, 5))
sns.heatmap(error_data, annot=True, fmt=".2f", cmap="Reds", cbar=True)
plt.title("Model Error Heatmap")
plt.xlabel("Question Index")
plt.ylabel("Error Type")
st.pyplot(plt)

# Download option
csv = df.to_csv(index=False)
st.download_button("📥 Download CSV", csv, "rag_comparison_results.csv", "text/csv")
