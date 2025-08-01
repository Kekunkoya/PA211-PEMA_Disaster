"""
Configuration file for Gemini API Project
"""

import os
from typing import Dict, Any

class GeminiConfig:
    """Configuration class for Gemini API settings"""
    
    # Default model settings
    DEFAULT_MODEL = 'gemini-pro'
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.8
    DEFAULT_TOP_K = 40
    DEFAULT_MAX_OUTPUT_TOKENS = 2048
    
    # Safety settings
    SAFETY_SETTINGS = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
    
    @classmethod
    def get_api_key(cls) -> str:
        """Get API key from environment variable"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it with: export GOOGLE_API_KEY='your-api-key-here'"
            )
        return api_key
    
    @classmethod
    def get_generation_config(cls, **kwargs) -> Dict[str, Any]:
        """Get generation configuration with optional overrides"""
        config = {
            'temperature': cls.DEFAULT_TEMPERATURE,
            'top_p': cls.DEFAULT_TOP_P,
            'top_k': cls.DEFAULT_TOP_K,
            'max_output_tokens': cls.DEFAULT_MAX_OUTPUT_TOKENS,
        }
        config.update(kwargs)
        return config
    
    @classmethod
    def get_safety_settings(cls) -> list:
        """Get safety settings"""
        return cls.SAFETY_SETTINGS.copy()

# Project-specific configurations
PROJECT_CONFIG = {
    'max_retries': 3,
    'timeout': 30,
    'log_level': 'INFO',
    'cache_responses': True,
    'cache_dir': './cache'
}

# Example usage configurations
EXAMPLE_CONFIGS = {
    'creative_writing': {
        'temperature': 0.9,
        'top_p': 0.9,
        'max_output_tokens': 1000
    },
    'code_generation': {
        'temperature': 0.2,
        'top_p': 0.8,
        'max_output_tokens': 2048
    },
    'analysis': {
        'temperature': 0.3,
        'top_p': 0.7,
        'max_output_tokens': 1500
    }
} 