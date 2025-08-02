import streamlit as st
import os
import fitz  # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai

# ---- CONFIG ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in environment variables.")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- FUNCTIONS ----
def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text("text") + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_embeddings(chunks):
    return embedder.encode(chunks, convert_to_tensor=False)

def retrieve_top_k(query, chunks, chunk_embeddings, k=1):
    query_emb = embedder.encode([query], convert_to_tensor=False)
    sims = cosine_similarity(query_emb, chunk_embeddings)[0]
    top_indices = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_openai_answer(query, context):
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_gemini_answer(query, context):
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def cosine_score(text1, text2):
    emb1 = embedder.encode([text1], convert_to_tensor=False)
    emb2 = embedder.encode([text2], convert_to_tensor=False)
    return cosine_similarity(emb1, emb2)[0][0]

# ---- STREAMLIT APP ----
st.title("📊 RAG Model Comparison: OpenAI vs Gemini")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Enter your query")
reference_answer = st.text_area("Reference answer (for scoring)", "")

if uploaded_file and query:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        chunk_embeddings = build_embeddings(chunks)
        top_context = " ".join(retrieve_top_k(query, chunks, chunk_embeddings, k=1))

    with st.spinner("Generating answers..."):
        openai_answer = generate_openai_answer(query, top_context)
        gemini_answer = generate_gemini_answer(query, top_context)

    st.subheader("🔹 OpenAI Answer")
    st.write(openai_answer)

    st.subheader("🔹 Gemini Answer")
    st.write(gemini_answer)

    if reference_answer.strip():
        openai_score = cosine_score(openai_answer, reference_answer)
        gemini_score = cosine_score(gemini_answer, reference_answer)
        model_to_model_score = cosine_score(openai_answer, gemini_answer)

        st.subheader("📈 Cosine Similarity Scores")
        st.write(f"OpenAI vs Reference: **{openai_score:.4f}**")
        st.write(f"Gemini vs Reference: **{gemini_score:.4f}**")
        st.write(f"OpenAI vs Gemini: **{model_to_model_score:.4f}**")
