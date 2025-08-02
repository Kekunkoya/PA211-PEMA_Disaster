import os
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import numpy as np
from bert_score import score as bert_score

# Initialize OpenAI client
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
top_k = st.number_input("Top K results", min_value=1, max_value=5, value=1)

if uploaded_files and query:
    # Load documents
    docs = load_pdfs(uploaded_files)
    
    # Get embeddings
    doc_embeddings = build_embeddings_openai(docs)
    query_embedding = build_embeddings_openai([query])[0]
    
    # Calculate similarities
    sims = [cosine_similarity(query_embedding, d) for d in doc_embeddings]
    
    # Get Top K docs
    top_indices = np.argsort(sims)[-top_k:][::-1]
    top_docs = [docs[i] for i in top_indices]
    
    st.subheader("Top Retrieved Document(s)")
    for i, doc in enumerate(top_docs):
        st.markdown(f"**Document {i+1} (Score: {sims[top_indices[i]]:.4f})**")
        st.write(doc[:1000] + "...")  # Show first 1000 chars
    
    # --- LLM Generation ---
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on retrieved documents."},
            {"role": "user", "content": f"Query: {query}\nContext: {top_docs[0]}"}
        ]
    )
    answer = completion.choices[0].message.content
    st.subheader("Generated Answer")
    st.write(answer)
    
    # --- BERTScore Evaluation ---
    P, R, F1 = bert_score([answer], [top_docs[0]], lang="en", verbose=False)
    st.subheader("Evaluation Metrics (BERTScore)")
    st.write(f"Precision: {P.mean().item():.4f}")
    st.write(f"Recall: {R.mean().item():.4f}")
    st.write(f"F1 Score: {F1.mean().item():.4f}")
