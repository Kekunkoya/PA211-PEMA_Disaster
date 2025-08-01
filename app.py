import os
from zipfile import ZipFile

# Create project folder structure
project_name = "rag_streamlit_app"
os.makedirs(project_name, exist_ok=True)

# Files content
app_py = """\
import streamlit as st
from rag_pipeline_openai import openai_rag
from rag_pipeline_gemini import gemini_rag
import os

st.set_page_config(page_title="RAG App - OpenAI & Gemini", layout="wide")

st.title("📚 RAG Demo App")
st.write("Compare OpenAI and Gemini answers side by side")

query = st.text_input("Enter your question:")

# Toggle RAG vs Direct LLM
mode = st.radio("Choose Mode:", ["RAG", "Direct LLM"])

# Model choice
model_choice = st.radio("Choose Model:", ["OpenAI", "Gemini", "Both"])

if st.button("Run"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        if model_choice in ["OpenAI", "Both"]:
            try:
                openai_answer = openai_rag(query, rag=(mode == "RAG"))
                st.subheader("OpenAI Answer")
                st.write(openai_answer)
            except Exception as e:
                st.error(f"OpenAI error: {e}")

        if model_choice in ["Gemini", "Both"]:
            try:
                from rag_pipeline_gemini import gemini_rag
                gemini_answer = gemini_rag(query, rag=(mode == "RAG"))
                st.subheader("Gemini Answer")
                st.write(gemini_answer)
            except Exception as e:
                st.error(f"Gemini error: {e}")
"""

rag_pipeline_openai_py = """\
import os
import faiss
import pickle
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INDEX_PATH = "docs/index.faiss"
DOCSTORE_PATH = "docs/docstore.pkl"

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(DOCSTORE_PATH, "rb") as f:
        docstore = pickle.load(f)
    return index, docstore

def retrieve_context(query, top_k=3):
    index, docstore = load_index()
    # Dummy retrieval simulation
    return ["Context chunk 1", "Context chunk 2", "Context chunk 3"]

def openai_rag(query, rag=True):
    if rag:
        context = retrieve_context(query)
        prompt = f"Answer the following based on context: {context}\\nQuestion: {query}"
    else:
        prompt = query

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message["content"]
"""

rag_pipeline_gemini_py = """\
import os
import faiss
import pickle
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_PATH = "docs/index.faiss"
DOCSTORE_PATH = "docs/docstore.pkl"

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(DOCSTORE_PATH, "rb") as f:
        docstore = pickle.load(f)
    return index, docstore

def retrieve_context(query, top_k=3):
    index, docstore = load_index()
    # Dummy retrieval simulation
    return ["Context chunk 1", "Context chunk 2", "Context chunk 3"]

def gemini_rag(query, rag=True):
    if rag:
        context = retrieve_context(query)
        prompt = f"Answer the following based on context: {context}\\nQuestion: {query}"
    else:
        prompt = query

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text
"""

requirements_txt = """\
streamlit
openai
google-generativeai
faiss-cpu
python-dotenv
PyPDF2
"""

env_template = """\
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
"""

# Create files
with open(os.path.join(project_name, "app.py"), "w") as f:
    f.write(app_py)

with open(os.path.join(project_name, "rag_pipeline_openai.py"), "w") as f:
    f.write(rag_pipeline_openai_py)

with open(os.path.join(project_name, "rag_pipeline_gemini.py"), "w") as f:
    f.write(rag_pipeline_gemini_py)

with open(os.path.join(project_name, "requirements.txt"), "w") as f:
    f.write(requirements_txt)

with open(os.path.join(project_name, ".env"), "w") as f:
    f.write(env_template)

# Create docs folder for FAISS index and docstore
os.makedirs(os.path.join(project_name, "docs"), exist_ok=True)

# Zip it
zip_path = f"/mnt/data/{project_name}.zip"
with ZipFile(zip_path, "w") as zipf:
    for root, dirs, files in os.walk(project_name):
        for file in files:
            filepath = os.path.join(root, file)
            zipf.write(filepath, arcname=os.path.relpath(filepath, project_name))

zip_path
