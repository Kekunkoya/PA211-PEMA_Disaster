# 🔑 Google API Key Setup Guide

## ❌ Current Issue
Your current API key is invalid. Here's how to get a proper one:

## 📋 Step-by-Step Instructions

### 1. Get Your API Key

1. **Visit Google AI Studio**: Go to [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

2. **Sign In**: Use your Google account to sign in

3. **Create API Key**:
   - Click the "Create API Key" button
   - You'll see a popup with your new API key
   - **Copy the entire key** (it should start with `AIza...`)

4. **Important Notes**:
   - The key should be about 39 characters long
   - It should start with `AIza`
   - Keep it secure and don't share it

### 2. Set Up the API Key

#### Option A: Quick Setup (Current Session Only)
```bash
export GOOGLE_API_KEY='your-actual-api-key-here'
```

#### Option B: Permanent Setup
```bash
echo 'export GOOGLE_API_KEY="your-actual-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

#### Option C: Use the Interactive Script
```bash
python fix_api_key.py
```

### 3. Test Your Setup

After setting the key, test it:
```bash
python test_gemini_setup.py
```

## 🔍 Troubleshooting

### Common Issues:

1. **"API key not valid"**
   - Make sure you copied the entire key
   - Check that it starts with `AIza`
   - Ensure no extra spaces or characters

2. **"API key not found"**
   - Make sure you set the environment variable
   - Check with: `echo $GOOGLE_API_KEY`

3. **"Model not found"**
   - This might be a regional issue
   - Try using `gemini-1.5-pro` instead of `gemini-pro`

### Valid API Key Format:
```
AIzaSyC1234567890abcdefghijklmnopqrstuvwxyz
```

## 🚀 Quick Test

Once you have your key, run this to test:

```bash
# Set your key (replace with your actual key)
export GOOGLE_API_KEY='AIzaSyC1234567890abcdefghijklmnopqrstuvwxyz'

# Test it
python -c "
import os
import google.generativeai as genai
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')
response = model.generate_content('Hello!')
print('✅ Success:', response.text)
"
```

## 📞 Need Help?

If you're still having issues:

1. **Check your Google account**: Make sure you're signed into the correct account
2. **Enable the API**: Go to [Google Cloud Console](https://console.cloud.google.com/) and enable the Generative Language API
3. **Check billing**: Some APIs require billing to be enabled
4. **Regional restrictions**: Make sure the API is available in your region

## 🎯 Next Steps

Once your API key is working:

1. **Test the setup**: `python test_gemini_setup.py`
2. **Run the main project**: `python gemini_project.py`
3. **Open Jupyter notebook**: `jupyter notebook gemini_notebook.ipynb`

---

**Remember**: Never commit your API key to version control or share it publicly! 