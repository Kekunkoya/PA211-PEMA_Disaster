"""
Gemini API Project Setup
This file demonstrates basic usage of Google's Gemini API
"""

import google.generativeai as genai
import os
from typing import Optional

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class GeminiProject:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini project with API key
        
        Args:
            api_key: Google AI API key. If None, will try to get from environment variable
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Set GOOGLE_API_KEY environment variable or pass it to constructor.")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-pro')
        
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using Gemini
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating text: {str(e)}"
    
    def chat_conversation(self, messages: list) -> str:
        """
        Have a conversation with Gemini
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Response from Gemini
        """
        try:
            chat = self.model.start_chat(history=[])
            response = chat.send_message(messages[-1]['content'])
            return response.text
        except Exception as e:
            return f"Error in chat conversation: {str(e)}"
    
    def analyze_document(self, document_text: str, analysis_prompt: str) -> str:
        """
        Analyze a document using Gemini
        
        Args:
            document_text: The document content to analyze
            analysis_prompt: Specific analysis instructions
            
        Returns:
            Analysis results
        """
        full_prompt = f"""
        Document to analyze:
        {document_text}
        
        Analysis instructions:
        {analysis_prompt}
        """
        
        return self.generate_text(full_prompt)

def main():
    """Example usage of the GeminiProject class"""
    
    # Example 1: Basic text generation
    print("=== Basic Text Generation ===")
    try:
        gemini = GeminiProject()
        
        prompt = "Explain quantum computing in simple terms"
        response = gemini.generate_text(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")
        
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Please set your GOOGLE_API_KEY environment variable")
    
    # Example 2: Document analysis
    print("=== Document Analysis ===")
    try:
        gemini = GeminiProject()
        
        document = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that work and react like humans. Some of the activities 
        computers with artificial intelligence are designed for include speech recognition, 
        learning, planning, and problem solving.
        """
        
        analysis_prompt = "Summarize the key points about AI mentioned in this document"
        analysis = gemini.analyze_document(document, analysis_prompt)
        print(f"Document: {document.strip()}")
        print(f"Analysis: {analysis}\n")
        
    except ValueError as e:
        print(f"Setup error: {e}")

if __name__ == "__main__":
    main() 