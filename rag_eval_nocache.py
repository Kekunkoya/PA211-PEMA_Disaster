import os
import streamlit as st
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import tempfile
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
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# ------------------ TEXT CHUNKING ------------------ #
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ------------------ FAISS RAG ------------------ #
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def retrieve_top_k(query, chunks, index, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k)
    return [chunks[i] for i in indices[0]]

# ------------------ LLM GENERATION ------------------ #
def generate_openai_answer(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_gemini_answer(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ------------------ COSINE SIMILARITY ------------------ #
def compute_cosine_similarity(text1, text2):
    embeddings = embedder.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# ------------------ STREAMLIT APP ------------------ #
st.title("📄 PDF RAG Evaluation – OpenAI vs Gemini")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Enter your query:")
reference_answer = st.text_area("Enter the reference answer (optional for scoring):", "")

if uploaded_pdf and query:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_pdf:
        tmp_pdf.write(uploaded_pdf.read())
        tmp_pdf_path = tmp_pdf.name

    pdf_text = extract_text_from_pdf(tmp_pdf_path)
    chunks = chunk_text(pdf_text)
    index, _ = build_faiss_index(chunks)

    top_context = " ".join(retrieve_top_k(query, chunks, index, k=3))
    openai_answer = generate_openai_answer(query, top_context)
    gemini_answer = generate_gemini_answer(query, top_context)

    st.subheader("Retrieved Context")
    st.write(top_context)

    st.subheader("Answers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**OpenAI GPT-4o-mini**")
        st.write(openai_answer)
    with col2:
        st.markdown("**Gemini 1.5 Flash**")
        st.write(gemini_answer)

    st.subheader("Similarity Score (Cosine)")
    score = compute_cosine_similarity(openai_answer, gemini_answer)
    st.write(f"Similarity between OpenAI and Gemini answers: **{score:.4f}**")

    if reference_answer:
        ref_score = compute_cosine_similarity(openai_answer, reference_answer)
        st.write(f"OpenAI vs Reference: **{ref_score:.4f}**")
        ref_score_g = compute_cosine_similarity(gemini_answer, reference_answer)
        st.write(f"Gemini vs Reference: **{ref_score_g:.4f}**")
