# build_dual_faiss_indexes.py

import os
import pickle
import faiss
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load data from data/ folder
DATA_DIR = "data"

def load_documents():
    loaders = [
        DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(DATA_DIR, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def build_and_save_faiss(docs, embeddings, index_path, store_path):
    # Build FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss.write_index(vectorstore.index, index_path)
    with open(store_path, "wb") as f:
        pickle.dump(vectorstore.docstore, f)
    print(f"✅ Saved FAISS index to {index_path} and docstore to {store_path}")

if __name__ == "__main__":
    # Load and split docs
    print("📂 Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    
    split_docs = split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")

    # --- Build OpenAI index ---
    print("🔹 Building OpenAI FAISS index...")
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    build_and_save_faiss(split_docs, openai_embeddings, "faiss_index_openai.idx", "docstore_openai.pkl")

    # --- Build Gemini index ---
    print("🔹 Building Gemini FAISS index...")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    build_and_save_faiss(split_docs, gemini_embeddings, "faiss_index_gemini.idx", "docstore_gemini.pkl")

    print("🎯 All indexes built successfully!")
