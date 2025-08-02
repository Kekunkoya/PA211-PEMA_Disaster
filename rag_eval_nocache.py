import streamlit as st
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
import nltk
nltk.download('punkt')

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
import google.generativeai as genai

# --- Load API Keys ---
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not openai_api_key or not gemini_api_key:
    st.error("Please set OPENAI_API_KEY and GEMINI_API_KEY in environment variables.")
    st.stop()

genai.configure(api_key=gemini_api_key)

# --- Embedding Builders ---
def build_embeddings_openai(texts):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings_model.embed_documents(texts)

def build_embeddings_gemini(texts):
    model = genai.GenerativeModel('models/text-embedding-004')
    vectors = []
    for t in texts:
        r = model.embed_content(content=t)
        vectors.append(r["embedding"])
    return vectors

# --- Retrieval Helper ---
def retrieve_top_k(query, docs, embeddings_fn, k=3):
    query_emb = embeddings_fn([query])[0]
    doc_embs = embeddings_fn([d.page_content for d in docs])
    sims = cosine_similarity([query_emb], doc_embs)[0]
    ranked = sorted(zip(docs, sims), key=lambda x: x[1], reverse=True)[:k]
    return ranked

# --- Evaluation ---
def evaluate_responses(reference, candidate, metrics):
    results = {}
    if "Cosine Similarity" in metrics:
        ref_vec = build_embeddings_openai([reference])[0]
        cand_vec = build_embeddings_openai([candidate])[0]
        results["Cosine"] = round(float(cosine_similarity([ref_vec], [cand_vec])[0][0]), 3)

    if "BERTScore" in metrics:
        P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
        results["BERT P"] = round(float(P[0]), 3)
        results["BERT R"] = round(float(R[0]), 3)
        results["BERT F1"] = round(float(F1[0]), 3)
        if "Error Score" in metrics:
            results["Error Score"] = round(1 - float(F1[0]), 3)

    if "Exact Match" in metrics:
        results["Exact Match"] = int(candidate.strip().lower() == reference.strip().lower())

    return results

# --- Streamlit UI ---
st.title("RAG Evaluation Without Cache")
st.write("Compare **OpenAI** vs **Gemini** results without using cached embeddings.")

uploaded_file = st.file_uploader("Upload Knowledge Base (PDF)", type=["pdf"])
if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    queries = st.text_area("Enter one or more queries (one per line)")
    top_k = st.slider("Top K retrieved chunks", 1, 5, 3)

    metrics = st.multiselect(
        "Select evaluation metrics",
        ["Cosine Similarity", "BERTScore", "Error Score", "Exact Match"],
        default=["Cosine Similarity", "BERTScore", "Error Score"]
    )

    if st.button("Run Evaluation"):
        results_table = []
        for q in queries.splitlines():
            if not q.strip():
                continue

            # Retrieve & Build Answers
            top_openai = retrieve_top_k(q, docs, build_embeddings_openai, k=top_k)
            top_gemini = retrieve_top_k(q, docs, build_embeddings_gemini, k=top_k)

            openai_answer = " ".join([doc.page_content for doc, _ in top_openai])
            gemini_answer = " ".join([doc.page_content for doc, _ in top_gemini])

            # Evaluation
            openai_scores = evaluate_responses(openai_answer, gemini_answer, metrics)

            results_table.append({
                "Query": q,
                "OpenAI Retrieved": openai_answer,
                "Gemini Retrieved": gemini_answer,
                **openai_scores
            })

        st.write("### Evaluation Results")
        st.dataframe(results_table)

else:
    st.info("Please upload a PDF knowledge base to start.")
