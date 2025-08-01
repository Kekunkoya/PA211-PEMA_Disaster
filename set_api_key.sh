#!/bin/bash

echo "🔑 Setting up Google API Key Environment"
echo "========================================"

# Check if API key is already set
if [ -n "$GOOGLE_API_KEY" ]; then
    echo "Current API key: ${GOOGLE_API_KEY:0:10}..."
else
    echo "No API key currently set"
fi

echo ""
echo "📝 To get your Google API Key:"
echo "1. Go to: https://makersuite.google.com/app/apikey"
echo "2. Sign in with your Google account"
echo "3. Click 'Create API Key'"
echo "4. Copy the key (should start with 'AIza...')"
echo ""

# Prompt for API key
read -p "Enter your Google API Key: " api_key

# Validate the API key format
if [[ $api_key == AIza* ]]; then
    # Set the environment variable
    export GOOGLE_API_KEY="$api_key"
    
    # Save to .env file
    echo "GOOGLE_API_KEY=$api_key" > .env
    
    # Add to shell profile for persistence
    echo "export GOOGLE_API_KEY=\"$api_key\"" >> ~/.zshrc
    
    echo ""
    echo "✅ API key set successfully!"
    echo "✅ Saved to .env file"
    echo "✅ Added to ~/.zshrc for persistence"
    echo ""
    echo "🔧 To activate in current session:"
    echo "   source ~/.zshrc"
    echo ""
    echo "🧪 To test the setup:"
    echo "   source rag_env/bin/activate"
    echo "   python test_imports.py"
else
    echo "❌ Invalid API key format. Should start with 'AIza'"
    exit 1
fi 