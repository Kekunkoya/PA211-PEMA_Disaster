import os
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# --- New imports for embeddings ---
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai

# --- For evaluation ---
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu

# --- Load environment variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not openai_api_key:
    st.warning("⚠️ OPENAI_API_KEY not set in environment.")
if not gemini_api_key:
    st.warning("⚠️ GEMINI_API_KEY not set in environment.")

# --- Configure Gemini ---
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)


# -------- Embedding Builders --------
def build_embeddings_openai(texts):
    """Generate embeddings using OpenAI."""
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        return embeddings_model.embed_documents(texts)
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding error: {e}")


def build_embeddings_gemini(texts):
    """Generate embeddings using Gemini."""
    try:
        model = "models/text-embedding-004"
        results = [genai.embed_content(model=model, content=txt)["embedding"] for txt in texts]
        return results
    except Exception as e:
        raise RuntimeError(f"Gemini embedding error: {e}")


# -------- Retrieval --------
def retrieve_top_k(query, docs, embeddings_fn, k=3):
    """Return top k most similar docs for a query."""
    query_emb = embeddings_fn([query])[0]
    doc_embs = embeddings_fn(docs)
    sims = cosine_similarity([query_emb], doc_embs)[0]
    ranked_idx = np.argsort(sims)[::-1][:k]
    return [(docs[i], sims[i]) for i in ranked_idx]


# -------- Evaluation --------
def evaluate_responses(reference, candidate):
    """Return BLEU and BERTScore for two responses."""
    try:
        bleu = sentence_bleu([reference.split()], candidate.split())
        P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
        return {
            "BLEU": round(bleu, 3),
            "BERTScore_P": round(P.item(), 3),
            "BERTScore_R": round(R.item(), 3),
            "BERTScore_F1": round(F1.item(), 3)
        }
    except Exception as e:
        return {"error": str(e)}


# -------- Streamlit UI --------
st.title("🔍 RAG Evaluation (No Cache)")
st.write("Compare OpenAI and Gemini RAG results without embedding caches.")

docs_input = st.text_area("Enter your documents (one per line):", height=150)
query_input = st.text_input("Enter your query:")
top_k = st.slider("Top K results:", 1, 5, 3)

if st.button("Run Evaluation"):
    if not docs_input.strip() or not query_input.strip():
        st.error("Please provide both documents and query.")
    else:
        docs = [d.strip() for d in docs_input.split("\n") if d.strip()]

        # Retrieve top results
        st.subheader("Top Retrieved Documents")
        try:
            top_openai = retrieve_top_k(query_input, docs, build_embeddings_openai, k=top_k)
            top_gemini = retrieve_top_k(query_input, docs, build_embeddings_gemini, k=top_k)

            st.write("**OpenAI:**")
            for doc, score in top_openai:
                st.write(f"- {doc} (score: {score:.3f})")

            st.write("**Gemini:**")
            for doc, score in top_gemini:
                st.write(f"- {doc} (score: {score:.3f})")
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            st.stop()

        # Evaluate only top-1
        if top_openai and top_gemini:
            reference = top_openai[0][0]
            openai_answer = top_openai[0][0]
            gemini_answer = top_gemini[0][0]

            st.subheader("Evaluation Metrics")
            st.write("Comparing top-1 result from each model against OpenAI top-1 as reference.")

            openai_scores = evaluate_responses(reference, openai_answer)
            gemini_scores = evaluate_responses(reference, gemini_answer)

            st.write("**OpenAI Scores:**", openai_scores)
            st.write("**Gemini Scores:**", gemini_scores)
