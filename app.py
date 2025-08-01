import streamlit as st
import os
from openai import OpenAI
import google.generativeai as genai
from rag_pipeline_openai import openai_rag
from rag_pipeline_gemini import gemini_rag

# --- Streamlit Config ---
st.set_page_config(page_title="PA 211 Disaster AI & RAG Demo", page_icon="📖", layout="wide")
st.title("📖 PA 211 Disaster AI & RAG Comparison App")

# --- Mode Selection ---
mode = st.selectbox(
    "Choose AI Mode:",
    [
        "OpenAI (Direct, no RAG)",
        "Gemini (Direct, no RAG)",
        "OpenAI with RAG",
        "Gemini with RAG"
    ]
)

# --- Input Query ---
query = st.text_input("Enter your question:")

# --- Direct OpenAI ---
def openai_direct(query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # Small fast model
        messages=[{"role": "user", "content": query}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# --- Direct Gemini ---
def gemini_direct(query):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(query)
    return resp.text

# --- Get Answer Button ---
if st.button("Get Answer"):
    if query.strip():
        try:
            with st.spinner("Generating answer..."):
                if mode == "OpenAI (Direct, no RAG)":
                    answer = openai_direct(query)
                elif mode == "Gemini (Direct, no RAG)":
                    answer = gemini_direct(query)
                elif mode == "OpenAI with RAG":
                    answer = openai_rag(query)
                elif mode == "Gemini with RAG":
                    answer = gemini_rag(query)

            st.subheader(f"Answer ({mode})")
            st.write(answer)

        except FileNotFoundError as e:
            st.error(f"Error: {e}")
            st.info("💡 Run `python build_dual_faiss_indexes.py` to build the missing FAISS indexes.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a question.")
