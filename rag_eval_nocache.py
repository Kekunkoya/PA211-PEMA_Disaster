import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import google.generativeai as genai

# ====== SETUP ======
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
    text = ""
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        pdf_doc = fitz.open(tmp_path)
        for page in pdf_doc:
            text += page.get_text("text") + "\n"
        pdf_doc.close()
        os.remove(tmp_path)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def retrieve_top_k(query, docs, k=1):
    doc_embeddings = embedder.encode(docs, convert_to_tensor=True)
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    top_results = np.argsort(-cos_scores)[:k]
    return [(docs[idx], float(cos_scores[idx])) for idx in top_results]

def generate_openai_answer(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_gemini_answer(query, context):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely:"
    response = model.generate_content(prompt)
    return response.text.strip()

# ====== STREAMLIT UI ======
st.title("RAG Evaluation: OpenAI vs Gemini")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query = st.text_input("Enter your question")

if uploaded_files and query:
    # Extract text and chunk
    with st.spinner("Extracting and chunking text..."):
        full_text = extract_text_from_pdfs(uploaded_files)
        chunks = chunk_text(full_text)

    # Retrieve top 1 chunk
    top_context, score = retrieve_top_k(query, chunks, k=1)[0]

    # Generate answers
    with st.spinner("Generating answers..."):
        openai_answer = generate_openai_answer(query, top_context)
        gemini_answer = generate_gemini_answer(query, top_context)

    # Display results
    st.subheader("Retrieved Context")
    st.write(top_context)

    st.subheader("Answers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**OpenAI Answer:**")
        st.write(openai_answer)
    with col2:
        st.markdown("**Gemini Answer:**")
        st.write(gemini_answer)

    st.caption(f"Cosine similarity score: {score:.4f}")
