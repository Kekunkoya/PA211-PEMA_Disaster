import os
import streamlit as st
import tempfile
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
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
def load_pdfs_and_split(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        with open(tmp_file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                    docs.extend(chunks)
    return docs

# ------------------ FAISS INDEX CREATION ------------------ #
def create_faiss_index(text_chunks):
    embeddings = embedder.encode(text_chunks, convert_to_tensor=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    return index, embeddings

# ------------------ RETRIEVAL ------------------ #
def retrieve_top_k(query, text_chunks, index, k=2):
    query_emb = embedder.encode([query], convert_to_tensor=False)
    D, I = index.search(np.array(query_emb, dtype="float32"), k)
    return [text_chunks[i] for i in I[0]]

# ------------------ LLM GENERATION ------------------ #
def generate_openai_answer(query, context):
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_gemini_answer(query, context):
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ------------------ COSINE SIMILARITY ------------------ #
def compute_cosine_similarity(text1, text2):
    embeddings = embedder.encode([text1, text2])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

# ------------------ STREAMLIT UI ------------------ #
st.title("📄 RAG Evaluation – OpenAI vs Gemini (FAISS, No Cache)")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
query = st.text_input("Enter your query:")
reference_answer = st.text_area("Enter the reference answer for scoring:")

if uploaded_files and query:
    # Step 1: Load documents
    docs = load_pdfs_and_split(uploaded_files)
    st.success(f"Loaded {len(docs)} text chunks from {len(uploaded_files)} PDF(s).")

    # Step 2: Create FAISS index
    index, _ = create_faiss_index(docs)

    # Step 3: Retrieve context
    top_contexts = retrieve_top_k(query, docs, index, k=2)
    combined_context = "\n\n".join(top_contexts)

    st.subheader("Retrieved Context")
    st.write(combined_context)

    # Step 4: Generate answers
    openai_answer = generate_openai_answer(query, combined_context)
    gemini_answer = generate_gemini_answer(query, combined_context)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**OpenAI GPT-4o-mini**")
        st.write(openai_answer)
    with col2:
        st.markdown("**Gemini 1.5 Flash**")
        st.write(gemini_answer)

    # Step 5: Compute similarity scores
    if reference_answer.strip():
        openai_score = compute_cosine_similarity(openai_answer, reference_answer)
        gemini_score = compute_cosine_similarity(gemini_answer, reference_answer)
        st.write(f"✅ OpenAI vs Reference: **{openai_score:.4f}**")
        st.write(f"✅ Gemini vs Reference: **{gemini_score:.4f}**")

    cross_score = compute_cosine_similarity(openai_answer, gemini_answer)
    st.write(f"🤝 OpenAI vs Gemini similarity: **{cross_score:.4f}**")
