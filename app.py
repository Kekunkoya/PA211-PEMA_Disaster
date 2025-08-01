import streamlit as st
from rag_pipeline_openai import openai_rag
from rag_pipeline_gemini import gemini_rag

st.set_page_config(page_title="PA 211 Disaster RAG Demo", page_icon="📖", layout="wide")

st.title("📖 PA 211 Disaster RAG Comparison App")

query = st.text_input("Enter your question:")

if st.button("Run RAG Search"):
    if query.strip():
        with st.spinner("Retrieving answers..."):
            openai_answer = openai_rag(query)
            gemini_answer = gemini_rag(query)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔵 OpenAI Answer")
            st.write(openai_answer)
        with col2:
            st.subheader("🟢 Gemini Answer")
            st.write(gemini_answer)
    else:
        st.warning("Please enter a question.")

