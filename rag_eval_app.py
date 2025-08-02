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
def load_notebook_outputs(path):
    """Extract text outputs from notebook cells."""
    nb = nbformat.read(path, as_version=4)
    outputs = []
    for cell in nb.cells:
        if cell.cell_type == "code" and "outputs" in cell:
            for output in cell.outputs:
                if output.output_type == "stream" and hasattr(output, "text"):
                    outputs.append(output.text.strip())
                elif output.output_type == "execute_result" and "text/plain" in output.data:
                    outputs.append(output.data["text/plain"].strip())
    return outputs

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

# Load notebook outputs
try:
    openai_outputs = load_notebook_outputs(NOTEBOOK_OPENAI)
    gemini_outputs = load_notebook_outputs(NOTEBOOK_GEMINI)
except FileNotFoundError as e:
    st.error(f"❌ Missing notebook file: {e}")
    st.stop()

# Prepare dataframe
df = pd.DataFrame({
    "Question": [item["question"] for item in pa211_data],
    "Reference Answer": [item["ideal_answer"] for item in pa211_data],
    "OpenAI Answer": openai_outputs[:len(pa211_data)],
    "Gemini Answer": gemini_outputs[:len(pa211_data)]
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
