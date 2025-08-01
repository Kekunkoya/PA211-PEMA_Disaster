# Gemini API Project

This project provides a comprehensive setup for using Google's Gemini API with Python. It includes both script-based and Jupyter notebook implementations for various AI tasks.

## 🚀 Quick Start

### 1. Virtual Environment Setup

The virtual environment is already set up in the `ISEM 770 GOOGLE` directory. To activate it:

```bash
source "ISEM 770 GOOGLE/bin/activate"
```

### 2. Install Dependencies

All required packages are already installed, including:
- `google-generativeai` - Google's official Gemini API client
- Other dependencies for RAG and data processing

### 3. Get Your API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Set it as an environment variable:

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

### 4. Test the Setup

Run the main project file:

```bash
python gemini_project.py
```

Or open the Jupyter notebook:

```bash
jupyter notebook gemini_notebook.ipynb
```

## 📁 Project Structure

```
├── gemini_project.py          # Main Python script with GeminiProject class
├── gemini_notebook.ipynb      # Interactive Jupyter notebook
├── config.py                  # Configuration settings
├── requirements.txt           # Project dependencies
├── GEMINI_README.md          # This file
└── ISEM 770 GOOGLE/          # Virtual environment directory
```

## 🔧 Features

### GeminiProject Class

The main `GeminiProject` class provides:

- **Text Generation**: Generate text responses to prompts
- **Chat Conversations**: Multi-turn conversations with context
- **Document Analysis**: Analyze and summarize documents
- **Error Handling**: Robust error handling and retry logic

### Configuration Management

The `config.py` file provides:

- **Default Settings**: Pre-configured generation parameters
- **Safety Settings**: Content filtering and safety controls
- **Custom Configurations**: Task-specific settings for different use cases

### Jupyter Notebook

The notebook includes:

- **Interactive Examples**: Live demonstrations of Gemini capabilities
- **Code Generation**: Generate and analyze Python code
- **Document Analysis**: Process and analyze text documents
- **Advanced Features**: Custom generation parameters and safety settings

## 📖 Usage Examples

### Basic Text Generation

```python
from gemini_project import GeminiProject

gemini = GeminiProject()
response = gemini.generate_text("Explain quantum computing")
print(response)
```

### Chat Conversation

```python
messages = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "user", "content": "Can you give me an example?"}
]

response = gemini.chat_conversation(messages)
print(response)
```

### Document Analysis

```python
document = "Your document text here..."
analysis_prompt = "Summarize the key points"

analysis = gemini.analyze_document(document, analysis_prompt)
print(analysis)
```

### Using Configuration

```python
from config import GeminiConfig, EXAMPLE_CONFIGS

# Get creative writing configuration
creative_config = EXAMPLE_CONFIGS['creative_writing']

# Use with custom settings
response = gemini.generate_text(
    "Write a creative story",
    **creative_config
)
```

## 🛠️ Advanced Features

### Custom Generation Parameters

```python
from config import GeminiConfig

# Get custom generation config
config = GeminiConfig.get_generation_config(
    temperature=0.9,
    max_output_tokens=1000
)

# Use with model
import google.generativeai as genai
model = genai.GenerativeModel(
    'gemini-pro',
    generation_config=config
)
```

### Safety Settings

The project includes pre-configured safety settings to filter harmful content:

- Harassment detection
- Hate speech filtering
- Explicit content blocking
- Dangerous content prevention

### Error Handling

The project includes robust error handling:

```python
def safe_generate_content(prompt, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return f"Failed after {max_retries} attempts"
```

## 🔍 Available Models

- **gemini-pro**: General-purpose text model
- **gemini-pro-vision**: Text and image understanding
- **gemini-1.5-pro**: Latest model with improved capabilities

## 📊 Use Cases

This project is designed for:

1. **Content Generation**: Articles, stories, summaries
2. **Code Generation**: Python scripts, functions, documentation
3. **Document Analysis**: Summarization, key point extraction
4. **Conversational AI**: Chatbots, virtual assistants
5. **Educational Content**: Explanations, tutorials, examples
6. **Creative Writing**: Stories, poems, creative content

## 🚨 Important Notes

1. **API Key Security**: Never commit your API key to version control
2. **Rate Limits**: Be aware of Google's API rate limits
3. **Content Safety**: The API includes safety filters by default
4. **Cost Management**: Monitor your API usage to manage costs

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GOOGLE_API_KEY` environment variable is set
2. **Import Error**: Make sure virtual environment is activated
3. **Rate Limit Error**: Implement retry logic with exponential backoff
4. **Content Filtered**: Adjust prompts to avoid triggering safety filters

### Getting Help

- Check the [Google AI Studio documentation](https://ai.google.dev/docs)
- Review the [google-generativeai Python library docs](https://github.com/google/generative-ai-python)
- Test with simple prompts first before complex tasks

## 📈 Next Steps

1. **Explore the Notebook**: Run through all examples in `gemini_notebook.ipynb`
2. **Customize Configurations**: Modify `config.py` for your specific needs
3. **Build Your Own**: Use the `GeminiProject` class as a foundation
4. **Integrate with RAG**: Combine with existing RAG notebooks for enhanced capabilities

## 🤝 Contributing

Feel free to extend this project with:

- Additional model configurations
- New use case examples
- Integration with other AI services
- Performance optimizations

---

**Happy coding with Gemini! 🚀** 