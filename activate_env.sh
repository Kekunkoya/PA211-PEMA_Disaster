#!/bin/bash
# Activation script for RAG Project virtual environment (Python 3.11 with PyTorch)

echo "Activating RAG Project virtual environment (Python 3.11)..."
source venv_py311/bin/activate
echo "✅ Virtual environment activated!"
echo "📦 Python packages installed:"
echo "   - openai, numpy, pandas, transformers"
echo "   - scikit-learn, networkx, matplotlib"
echo "   - spacy, faiss-cpu, rank-bm25"
echo "   - torch 2.7.1 (PyTorch for ML models)"
echo "   - and many more RAG-related packages"
echo ""
echo "To deactivate, run: deactivate"
echo "To start Jupyter, run: jupyter notebook" 