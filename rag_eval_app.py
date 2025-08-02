import streamlit as st
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai

# ---------------- CONFIG ---------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCS_FOLDER = "data"  # <-- Change to your folder path

if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY in environment variables.")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in environment variables.")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------- LOAD DOCUMENTS --------------- #
def load_documents_from_folder(folder_path):
    """Load all .txt, .md, and .docx files from a folder."""
    docs = []
    for filepath in glob.glob(os.path.join(folder_path, "**"), recursive=True):
        if os.path.isfile(filepath) and filepath.lower().endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())
        # Optional: handle PDFs here if needed
    return docs

# --------------- RETRIEVAL --------------- #
def retrieve_top_k(query, docs, k=2):
    """Retrieve top-k most relevant chunks from docs."""
    doc_embeddings = embedder.encode(docs)
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return [docs[i] for i in top_indices]

# --------------- LLM GENERATION --------------- #
def generate_openai_answer(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API Error: {e}"

def generate_gemini_answer(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini API Error: {e}"

# --------------- COSINE SIMILARITY --------------- #
def compute_cosine_similarity(text1, text2):
    emb = embedder.encode([text1, text2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

# --------------- STREAMLIT APP --------------- #
st.title("📄 Folder-based RAG Evaluation – OpenAI vs Gemini (No Cache)")

query = st.text_input("Enter your query:")
reference_answer = st.text_area("Enter reference answer (optional for scoring):")

if query:
    with st.spinner("Loading documents..."):
        docs = load_documents_from_folder(DOCS_FOLDER)
        if not docs:
            st.error(f"No documents found in {DOCS_FOLDER}. Please add some .txt or .md files.")
            st.stop()

    with st.spinner("Retrieving top context..."):
        top_context = " ".join(retrieve_top_k(query, docs, k=2))

    with st.spinner("Generating answers..."):
        openai_answer = generate_openai_answer(query, top_context)
        gemini_answer = generate_gemini_answer(query, top_context)

    st.subheader("📌 Retrieved Context")
    st.write(top_context)

    st.subheader("📝 Answer Comparison")
    st.table({
        "Model": ["OpenAI", "Gemini"],
        "Answer": [openai_answer, gemini_answer]
    })

    if reference_answer:
        st.subheader("📊 Similarity Scores")
        openai_score = compute_cosine_similarity(openai_answer, reference_answer)
        gemini_score = compute_cosine_similarity(gemini_answer, reference_answer)
        cross_score = compute_cosine_similarity(openai_answer, gemini_answer)

        st.table({
            "Comparison": ["OpenAI vs Reference", "Gemini vs Reference", "OpenAI vs Gemini"],
            "Cosine Similarity": [f"{openai_score:.4f}", f"{gemini_score:.4f}", f"{cross_score:.4f}"]
        })
