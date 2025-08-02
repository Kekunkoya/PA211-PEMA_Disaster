import os
import streamlit as st
import openai
import numpy as np
from PyPDF2 import PdfReader  # Replacing fitz
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
import openai
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# PDF text extraction
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Chunk text into fixed size
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Build embeddings
def build_embeddings(text_chunks):
    return embedder.encode(text_chunks, convert_to_tensor=True)

# Retrieve top K chunk
def retrieve_top_k(query, chunks, chunk_embeddings, k=1):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=k)[0]
    return [chunks[hit['corpus_id']] for hit in hits]

# LLM generation
def generate_openai(prompt):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message["content"]

def generate_gemini(prompt):
    model = genai.GenerativeModel("gemini-pro")
    resp = model.generate_content(prompt)
    return resp.text

# Evaluation metrics
def evaluate(candidate, reference):
    cosine_sim = util.cos_sim(embedder.encode(candidate), embedder.encode(reference)).item()
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
    return cosine_sim, P[0].item(), R[0].item(), F1[0].item()

# Streamlit UI
st.title("📄 RAG Evaluation App (PDF Upload)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Enter your question/query:")

if uploaded_file and query:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text, chunk_size=500)
        chunk_embeddings = build_embeddings(chunks)

    with st.spinner("Retrieving top chunk..."):
        top_chunk = retrieve_top_k(query, chunks, chunk_embeddings, k=1)[0]

    st.subheader("Retrieved Context")
    st.write(top_chunk)

    with st.spinner("Generating answers..."):
        openai_answer = generate_openai(f"Answer the question based on the context:\n{top_chunk}\n\nQuestion: {query}")
        gemini_answer = generate_gemini(f"Answer the question based on the context:\n{top_chunk}\n\nQuestion: {query}")

    st.subheader("Model Answers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**OpenAI Response:**")
        st.write(openai_answer)
    with col2:
        st.markdown("**Gemini Response:**")
        st.write(gemini_answer)

    with st.spinner("Evaluating..."):
        cos_openai, p_openai, r_openai, f1_openai = evaluate(openai_answer, top_chunk)
        cos_gemini, p_gemini, r_gemini, f1_gemini = evaluate(gemini_answer, top_chunk)

    st.subheader("Evaluation Results")
    st.markdown("### OpenAI")
    st.write(f"Cosine Similarity: {cos_openai:.4f}")
    st.write(f"BERTScore - Precision: {p_openai:.4f}, Recall: {r_openai:.4f}, F1: {f1_openai:.4f}")

    st.markdown("### Gemini")
    st.write(f"Cosine Similarity: {cos_gemini:.4f}")
    st.write(f"BERTScore - Precision: {p_gemini:.4f}, Recall: {r_gemini:.4f}, F1: {f1_gemini:.4f}")
