streamlit as st
import os
import openai
import google.generativeai as genai

# --- Load API Keys ---
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if OPENAI_KEY:
    openai.api_key = OPENAI_KEY
else:
    st.warning("⚠️ OpenAI API key not found in environment.")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    st.warning("⚠️ Gemini API key not found in environment.")

# --- Streamlit UI ---
st.title("🔍 RAG Demo - OpenAI & Gemini Toggle (Fail‑Safe)")
st.write("Ask a question — the app will use the selected model.")

model_choice = st.selectbox("Choose Model", ["OpenAI (GPT-4o-mini)", "Gemini (Pro)", "Gemini (2.0 Flash)"])
query = st.text_area("Enter your question:")

if st.button("Get Answer"):
    if not query.strip():
        st.error("Please enter a question.")
    else:
        if model_choice.startswith("OpenAI"):
            with st.spinner("Querying OpenAI..."):
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": query}]
                    )
                    st.success("✅ OpenAI Response")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"❌ OpenAI Error: {e}")

        else:
            model_name = "models/gemini-pro" if "Pro" in model_choice else "models/gemini-2.0-flash"
            with st.spinner(f"Querying Gemini ({model_name})..."):
                try:
                    model = genai.GenerativeModel(model_name)
                    resp = model.generate_content(query)
                    st.success(f"✅ Gemini Response ({model_name})")
                    st.write(resp.text)
                except Exception as e:
                    st.error(f"❌ Gemini Error: {e}")
                    st.info("🔄 Falling back to OpenAI...")
                    try:
                        response = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": query}]
                        )
                        st.success("✅ OpenAI Fallback Response")
                        st.write(response.choices[0].message.content)
                    except Exception as oe:
                        st.error(f"❌ OpenAI Fallback Error: {oe}")

# --- Optional: Embedding Function ---
def get_embedding(text, provider="gemini"):
    if provider == "gemini" and GEMINI_KEY:
        try:
            emb = genai.embed_content(
                model="models/text-embedding-001",
                content=text
            )
            return emb['embedding']
        except Exception as e:
            st.error(f"Gemini Embedding Error: {e}")
    elif provider == "openai" and OPENAI_KEY:
        try:
            emb = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return emb.data[0].embedding
        except Exception as e:
            st.error(f"OpenAI Embedding Error: {e}")
    return None
