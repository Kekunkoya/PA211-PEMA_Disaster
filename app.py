import os
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline_openai import openai_rag
from rag_pipeline_gemini import gemini_rag
from openai import OpenAI
import google.generativeai as genai

# Load API keys from .env
load_dotenv()

st.set_page_config(page_title="PA 211 Disaster AI & RAG Demo", page_icon="📖", layout="wide")
st.title("📖 PA 211 Disaster AI & RAG Comparison App")

# Toggle for Mode
mode = st.selectbox(
    "Choose AI Mode:",
    [
        "OpenAI (Direct, no RAG)",
        "Gemini (Direct, no RAG)",
        "OpenAI with RAG",
        "Gemini with RAG"
    ]
)

query = st.text_input("Enter your question:")

# --- Direct OpenAI ---
def openai_direct(query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
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

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Generating answer..."):
            try:
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
                st.error(f"File not found: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
