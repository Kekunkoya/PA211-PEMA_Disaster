import os
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import numpy as np
from bert_score import score as bert_score

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Build embeddings using OpenAI ---
def build_embeddings_openai(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [np.array(e.embedding) for e in response.data]

# --- Cosine similarity ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- PDF text extraction ---
def load_pdfs(uploaded_files):
    docs = []
    for file in uploaded_files:
        pdf = PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        docs.append(text)
    return docs

# --- Streamlit App ---
st.title("RAG Evaluation (No Cache)")
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
query = st.text_input("Enter your query")

if uploaded_files and query:
    docs = load_pdfs(uploaded_files)
    doc_embeddings = build_embeddings_openai(docs)
    query_embedding = build_embeddings_openai([query])[0]
    
    sims = [cosine_similarity(query_embedding, d) for d in doc_embeddings]
    top_doc_idx = np.argmax(sims)
    top_doc = docs[top_doc_idx]
    
    st.subheader("Top Retrieved Document")
    st.write(top_doc)

    # --- LLM Generation ---
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on retrieved documents."},
            {"role": "user", "content": f"Query: {query}\nContext: {top_doc}"}
        ]
    )
    answer = completion.choices[0].message.content
    st.subheader("Generated Answer")
    st.write(answer)

    # --- BERTScore Evaluation ---
    P, R, F1 = bert_score([answer], [top_doc], lang="en", verbose=False)
    st.subheader("Evaluation Metrics")
    st.write(f"Precision: {P.mean().item():.4f}")
    st.write(f"Recall: {R.mean().
