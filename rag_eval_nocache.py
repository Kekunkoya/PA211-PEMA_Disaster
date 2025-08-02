import streamlit as st
import os
import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai

# ------------------ CONFIG ------------------ #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY in environment variables.")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in environment variables.")

# Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ PDF LOADER ------------------ #
def extract_text_from_pdf(uploaded_file):
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ------------------ RETRIEVAL ------------------ #
def retrieve_top_k(query, docs, k=1):
    """Retrieve top-k most relevant document chunks."""
    doc_embeddings = embedder.encode(docs)
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [docs[i] for i in top_indices]

# ------------------ LLM GENERATION ------------------ #
def generate_openai_answer(query, context):
    """Generate an answer using OpenAI."""
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_gemini_answer(query, context):
    """Generate an answer using Gemini 1.5."""
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ------------------ COSINE SIMILARITY ------------------ #
def compute_cosine_similarity(text1, text2):
    """Compute cosine similarity between two texts."""
    embeddings = embedder.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# ------------------ STREAMLIT APP ------------------ #
st.title("📄 RAG Evaluation – OpenAI vs Gemini (No Cache)")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Enter your query:")
reference_answer = st.text_area("Enter the reference answer (optional for scoring):", "")

if uploaded_pdf and query:
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    docs = pdf_text.split("\n\n")  # split into paragraphs

    # Retrieve context
    top_context = " ".join(retrieve_top
