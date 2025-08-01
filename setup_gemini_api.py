#!/usr/bin/env python3
"""
Simple script to set up Google API key for Gemini
"""

import os
from pathlib import Path

def test_api_key(api_key):
    """Test if the API key is valid"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content('Hello!')
        return True
    except Exception:
        return False

def setup_gemini_api():
    """Set up Google API key for Gemini"""
    print("🔧 Setting up Google API Key for Gemini")
    print("=" * 50)
    
    # Check if API key is already set and test it
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print(f"🔍 Testing existing API key: {api_key[:10]}...")
        if test_api_key(api_key):
            print("✅ Existing API key is valid!")
            return True
        else:
            print("❌ Existing API key is invalid. Please enter a new one.")
    
    print("📝 To get your Google API Key:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the key (should start with 'AIza...')")
    print()
    
    api_key = input("Enter your Google API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided.")
        return False
    
    if not api_key.startswith('AIza'):
        print("❌ Invalid API key format. Should start with 'AIza'")
        return False
    
    # Test the new API key
    print("🧪 Testing your API key...")
    if not test_api_key(api_key):
        print("❌ API key test failed. Please check your key and try again.")
        return False
    
    # Create .env file
    env_file = Path('.env')
    with open(env_file, 'w') as f:
        f.write(f"GOOGLE_API_KEY={api_key}\n")
    
    print(f"✅ Google API Key saved to {env_file}")
    
    # Set environment variable for current session
    os.environ['GOOGLE_API_KEY'] = api_key
    
    return True

def test_gemini_connection():
    """Test the Gemini API connection"""
    print("\n🧪 Testing Gemini API Connection")
    print("=" * 40)
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ API key not found in environment")
            return False
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Test with a simple request
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content('Hello! Please respond with "API working!"')
        
        print("✅ Gemini API connection successful!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        return False

def main():
    """Main function"""
    print("🎯 Gemini API Setup for RAG Project")
    print("=" * 50)
    
    # Setup API key
    if not setup_gemini_api():
        print("\n❌ Setup failed. Please try again.")
        return
    
    # Test connection
    if test_gemini_connection():
        print("\n🎉 Setup complete! Your Gemini API is ready to use.")
        print("\nYou can now:")
        print("- Run your RAG notebooks")
        print("- Use Google Generative AI in your scripts")
        print("- Test with: python test_imports.py")
    else:
        print("\n❌ API test failed. Please check your API key.")

if __name__ == "__main__":
    main() 