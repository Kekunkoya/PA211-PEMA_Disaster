# 🔑 Google API Key Setup Guide

## Quick Setup Options

### Option 1: Use the Setup Script (Recommended)
```bash
./set_api_key.sh
```

### Option 2: Manual Setup

#### Step 1: Get Your API Key
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (should start with `AIza`)

#### Step 2: Set the Environment Variable

**For current session only:**
```bash
export GOOGLE_API_KEY="your-actual-api-key-here"
```

**For permanent setup:**
```bash
echo 'export GOOGLE_API_KEY="your-actual-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

**Using .env file:**
```bash
echo "GOOGLE_API_KEY=your-actual-api-key-here" > .env
```

#### Step 3: Test Your Setup
```bash
source rag_env/bin/activate
python test_imports.py
```

## API Key Format
- Should start with `AIza`
- About 39 characters long
- Example: `AIzaSyC1234567890abcdefghijklmnopqrstuvwxyz`

## Troubleshooting

### "API key not valid"
- Make sure you copied the entire key
- Check that it starts with `AIza`
- Ensure no extra spaces or characters

### "API key not found"
- Make sure you set the environment variable
- Check with: `echo $GOOGLE_API_KEY`

### "Model not found"
- Try using `gemini-1.5-flash` instead of `gemini-pro`

## Next Steps
Once your API key is working:
1. Test with: `python test_imports.py`
2. Run your RAG notebooks
3. Start developing your RAG application 