import os
import streamlit as st
import tempfile
import PyPDF2
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# ==============================
# CONFIGURE API KEYS
# ==============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
if not GEMINI_API_KEY:
    st.error("Please set the GEMINI_API_KEY environment variable.")

# OpenAI & Gemini Setup
client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Sentence Transformer model for similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ==============================
# PDF LOADER & EMBEDDING
# ==============================
def load_pdfs_and_split(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                    docs.extend(chunks)
    return docs

def build_embeddings(texts):
    return embedder.encode(texts, convert_to_tensor=True)

# ==============================
# RETRIEVAL FUNCTION
# ==============================
def retrieve_top_k(query, docs, doc_embeddings, k=1):
    query_emb = embedder.encode([query], convert_to_tensor=True)
    hits = util.semantic_search(query_emb, doc_embeddings, top_k=k)[0]
    return [(docs[hit['corpus_id']], hit['score']) for hit in hits]

# ==============================
# LLM GENERATION
# ==============================
def generate_openai_answer(query, context):
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def generate_gemini_answer(query, context):
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"
    response = gemini_model.generate_content(prompt)
    return response.text

# ==============================
# STREAMLIT UI
# ==============================
st.title("📄 RAG Evaluation (OpenAI vs Gemini 1.5)")
st.write("Upload PDFs, enter a query, and compare OpenAI vs Gemini responses side-by-side.")

uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    docs = load_pdfs_and_split(uploaded_files)
    st.success(f"Loaded {len(docs)} chunks from {len(uploaded_files)} PDFs.")
    doc_embeddings = build_embeddings(docs)

    query = st.text_input("Enter your question")
    top_k = st.number_input("Top K passages to retrieve", min_value=1, max_value=5, value=1)

    if st.button("Run RAG Evaluation"):
        if query.strip() == "":
            st.error("Please enter a question.")
        else:
            top_contexts = retrieve_top_k(query, docs, doc_embeddings, k=top_k)
            combined_context = "\n\n".join([ctx for ctx, _ in top_contexts])

            st.subheader("Retrieved Context")
            st.write(combined_context)

            openai_answer = generate_openai_answer(query, combined_context)
            gemini_answer = generate_gemini_answer(query, combined_context)

            st.subheader("Answers")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**OpenAI GPT-4o-mini**")
                st.write(openai_answer)
            with col2:
                st.markdown("**Gemini 1.5 Flash**")
                st.write(gemini_answer)

            st.subheader("Similarity Score (Cosine)")
            openai_emb = embedder.encode([openai_answer], convert_to_tensor=True)
            gemini_emb = embedder.encode([gemini_answer], convert_to_tensor=True)
            score = util.cos_sim(openai_emb, gemini_emb).item()
            st.write(f"Similarity between OpenAI and Gemini answers: **{score:.4f}**")
