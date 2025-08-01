import streamlit as st
import os

st.set_page_config(page_title="RAG Demo", page_icon="📚", layout="wide")
st.title("📚 RAG Demo - Toggle Between OpenAI & Gemini")

# --- Select model provider ---
provider = st.selectbox("Choose LLM Provider", ["OpenAI", "Gemini"])

# --- OpenAI Option ---
if provider == "OpenAI":
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_answer(prompt):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # Change to "gpt-4o" if you have access
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

# --- Gemini Option ---
elif provider == "Gemini":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_answer(prompt):
        model = genai.GenerativeModel("gemini-pro")
        resp = model.generate_content(prompt)
        return resp.text

# --- Query Input ---
query = st.text_input("Ask a question or enter your prompt:")

# --- Display Answer ---
if query:
    with st.spinner("Generating answer..."):
        st.write(generate_answer(query))

st.markdown("---")
st.caption("Powered by OpenAI & Google Gemini - Streamlit RAG Demo")
