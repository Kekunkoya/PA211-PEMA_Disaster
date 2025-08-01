# RAG Streamlit Demo with OpenAI & Gemini Toggle

This project is a **Retrieval-Augmented Generation (RAG)** demo built with **Streamlit**, allowing you to toggle between **OpenAI GPT models** and **Google Gemini models** for answering questions.


**A Retrieval-Augmented Generation (RAG) Streamlit app to support disaster response using PA 211 services.**

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-blueviolet)
![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


---

## Features
- **Toggle between OpenAI and Gemini** in real time.
- Load and query documents for retrieval-augmented generation.
- Simple UI built with Streamlit.
- Ready for deployment on **Streamlit Cloud**.
- Works locally on your machine.

---
## Directory Structure

```
📦 pa211-disaster-rag/
├── app/
│   └── streamlit_app.py
├── data/
│   └── <PDF, DOCX, RTF disaster docs>
├── embeddings/
├── logs/
├── src/
│   ├── embed_documents.py
│   ├── feedback.py
│   ├── generation.py
│   ├── query_transform.py
│   └── retrieval.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

