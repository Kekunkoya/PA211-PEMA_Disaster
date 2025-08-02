
import streamlit as st
import os
from openai_rag_combined import main as openai_main, build_cache as openai_build_cache
from gemini_rag_combined import main as gemini_main, build_cache as gemini_build_cache

# --- Streamlit App ---
st.set_page_config(page_title="PA 211 RAG Assistant", layout="centered")

st.title("PA 211 RAG Assistant")
st.markdown("Ask questions about PA 211 resources and get answers from OpenAI or Gemini with retrieval-augmented generation.")

# API Keys
st.sidebar.header("API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

# Model Selector
api_choice = st.radio("Choose API", ["OpenAI", "Gemini"])

# Retrieval settings
top_k = st.slider("Number of retrieved documents (Top K)", min_value=1, max_value=10, value=3)

# Cache build button
if st.button("Build Embedding Cache"):
    if api_choice == "OpenAI":
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please provide your OpenAI API Key in the sidebar.")
        else:
            with st.spinner("Building embedding cache for OpenAI..."):
                openai_build_cache()
            st.success("OpenAI embedding cache built successfully!")
    elif api_choice == "Gemini":
        if not os.getenv("GEMINI_API_KEY"):
            st.error("Please provide your Gemini API Key in the sidebar.")
        else:
            with st.spinner("Building embedding cache for Gemini..."):
                gemini_build_cache()
            st.success("Gemini embedding cache built successfully!")

# Query input
query = st.text_area("Enter your query")

if st.button("Get Answer"):
    if api_choice == "OpenAI":
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please provide your OpenAI API Key in the sidebar.")
        else:
            with st.spinner("Retrieving and generating answer from OpenAI..."):
                answer = openai_main(query, top_k=top_k)
                st.success(answer)
    elif api_choice == "Gemini":
        if not os.getenv("GEMINI_API_KEY"):
            st.error("Please provide your Gemini API Key in the sidebar.")
        else:
            with st.spinner("Retrieving and generating answer from Gemini..."):
                answer = gemini_main(query, top_k=top_k)
                st.success(answer)
