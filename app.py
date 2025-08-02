import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not openai_api_key:
    st.warning("⚠️ OPENAI_API_KEY not set.")
if not gemini_api_key:
    st.warning("⚠️ GEMINI_API_KEY not set.")

# Initialize clients
openai_client = OpenAI(api_key=openai_api_key)
genai.configure(api_key=gemini_api_key)

# --------------------------
# UTILS
# --------------------------
def load_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def save_embeddings_cache(embeddings, filename):
    """Save embeddings to a pickle file for reuse."""
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)

def build_embeddings_openai(texts):
    try:
        response = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
        embeddings = [item.embedding for item in response.data]
        return embeddings
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding error: {e}")

def build_embeddings_gemini(texts):
    try:
        model = genai.GenerativeModel("models/text-embedding-004")  # Gemini 1.5 embedding model
        embeddings = [model.embed_content(content=text)["embedding"] for text in texts]
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Gemini embedding error: {e}")

def create_vectorstore(chunks, embeddings, api_choice):
    documents = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_embeddings(list(zip(documents, embeddings)))
    vectorstore.save_local(f"faiss_index_{api_choice.lower()}")
    return vectorstore

# --------------------------
# STREAMLIT APP
# --------------------------
st.title("📄 PA 211 RAG App with OpenAI & Gemini")
st.write("Upload a PDF, choose your embedding API, and run RAG.")

# Sidebar options
api_choice = st.sidebar.radio("Choose API for embeddings:", ["OpenAI", "Gemini"])

uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
query = st.text_input("Enter your question:")

if uploaded_pdf:
    with st.spinner("Reading and splitting PDF..."):
        text = load_pdf_text(uploaded_pdf)
        chunks = split_text_into_chunks(text)

    if st.button("Generate Embeddings & Save Cache"):
        with st.spinner(f"Generating {api_choice} embeddings..."):
            if api_choice == "OpenAI":
                embeddings = build_embeddings_openai(chunks)
                save_embeddings_cache(embeddings, "openai_cache.pkl")
            else:
                embeddings = build_embeddings_gemini(chunks)
                save_embeddings_cache(embeddings, "gemini_cache.pkl")

            st.success(f"{api_choice} embeddings created and cached.")
            create_vectorstore(chunks, embeddings, api_choice)

# --------------------------
# RAG Search
# --------------------------
if query:
    with st.spinner("Searching vectorstore..."):
        index_path = f"faiss_index_{api_choice.lower()}"
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(index_path, embeddings=None, allow_dangerous_deserialization=True)
            docs = vectorstore.similarity_search(query, k=3)
            st.subheader("Top Retrieved Chunks")
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Result {i}:** {doc.page_content}")
        else:
            st.error("No FAISS index found. Please generate embeddings first.")
