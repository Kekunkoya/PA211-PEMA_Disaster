import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import google.generativeai as genai

# ====== SETUP ======
st.set_page_config(page_title="RAG Evaluation - OpenAI vs Gemini", layout="wide")

openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)

if not openai_api_key or not gemini_api_key:
    st.error("Please set your OPENAI_API_KEY and GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=openai_api_key)
genai.configure(api_key=gemini_api_key)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ====== FUNCTIONS ======
def extract_text_from_pdfs(uploaded_files):
    """Extract text from uploaded PDFs."""
    text = ""
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        pdf_doc = PdfReader(tmp_path)
        for page in pdf_doc.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        os.remove(tmp_path)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def retrieve_top_k(query, docs, k=1):
    """Retrieve top-k most relevant chunks using cosine similarity."""
    doc_embeddings = embedder.encode(docs, convert_to_tensor=True)
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    top_results = np.argsort(-cos_scores)[:k]
    return [(docs[idx], float(cos_scores[idx])) for idx in top_results]

def generate_openai_answer(query, context):
    """Generate an answer using OpenAI GPT model."""
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_gemini_answer(query, context):
    """Generate an answer using Gemini model."""
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely:"
    response = model.generate_content(prompt)
    return response.text.strip()

# ====== STREAMLIT UI ======
st.title("📄 RAG Evaluation: OpenAI vs Gemini")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query_mode = st.radio("Select mode:", ["Single Question", "Batch Questions"])

if uploaded_files:
    with st.spinner("Extracting and chunking text..."):
        full_text = extract_text_from_pdfs(uploaded_files)
        chunks = chunk_text(full_text)

    if query_mode == "Single Question":
        query = st.text_input("Enter your question")
        if query:
            top_context, score = retrieve_top_k(query, chunks, k=1)[0]
            with st.spinner("Generating answers..."):
                openai_answer = generate_openai_answer(query, top_context)
                gemini_answer = generate_gemini_answer(query, top_context)

            st.subheader("Retrieved Context")
            st.write(top_context)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**OpenAI Answer:**")
                st.write(openai_answer)
            with col2:
                st.markdown("**Gemini Answer:**")
                st.write(gemini_answer)

            st.caption(f"Cosine similarity score: {score:.4f}")

    else:
        questions_file = st.file_uploader("Upload a TXT file with one question per line", type=["txt"])
        if questions_file:
            questions = [q.strip() for q in questions_file.read().decode("utf-8").splitlines() if q.strip()]
            results = []
            for q in questions:
                top_context, score = retrieve_top_k(q, chunks, k=1)[0]
                openai_answer = generate_openai_answer(q, top_context)
                gemini_answer = generate_gemini_answer(q, top_context)
                results.append((q, openai_answer, gemini_answer, score))

            st.subheader("Batch Results")
            for q, oa, ga, sc in results:
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**OpenAI:** {oa}")
                st.markdown(f"**Gemini:** {ga}")
                st.caption(f"Cosine similarity: {sc:.4f}")
                st.markdown("---")
