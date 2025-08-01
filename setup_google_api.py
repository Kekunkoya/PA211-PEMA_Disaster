#!/usr/bin/env python3
"""
Setup script for ISEM 770 GOOGLE Project
This script helps you configure your Google API key and test the environment.
"""

import os
import sys
from pathlib import Path

def setup_google_api_key():
    """Set up Google API key for the project."""
    print("🔧 Setting up Google API Key for ISEM 770 GOOGLE Project")
    print("=" * 60)
    
    # Check if API key is already set
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print(f"✅ Google API Key is already set: {api_key[:10]}...")
        return True
    
    print("📝 Please enter your Google API Key:")
    print("   You can get one from: https://console.cloud.google.com/")
    print("   - Go to APIs & Services > Credentials")
    print("   - Create a new API Key")
    print()
    
    api_key = input("Enter your Google API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Please run this script again with a valid API key.")
        return False
    
    # Create .env file
    env_file = Path('.env')
    with open(env_file, 'w') as f:
        f.write(f"GOOGLE_API_KEY={api_key}\n")
    
    print(f"✅ Google API Key saved to {env_file}")
    print("🔒 The .env file has been added to .gitignore for security")
    
    # Set environment variable for current session
    os.environ['GOOGLE_API_KEY'] = api_key
    
    return True

def test_environment():
    """Test if the environment is properly set up."""
    print("\n🧪 Testing Environment Setup")
    print("=" * 40)
    
    # Test Python packages
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'openai', 'requests',
        'google_api_python_client', 'google_auth', 'jupyter'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Test Google API
    try:
        from googleapiclient.discovery import build
        print("✅ Google API Python Client")
    except ImportError:
        print("❌ Google API Python Client - NOT FOUND")
        return False
    
    # Test API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print("✅ Google API Key is set")
        return True
    else:
        print("❌ Google API Key is not set")
        return False

def create_startup_script():
    """Create a startup script to activate the environment."""
    script_content = '''#!/bin/bash
# Startup script for ISEM 770 GOOGLE Project

echo "🚀 Starting ISEM 770 GOOGLE Project Environment"
echo "================================================"

# Activate virtual environment
source "ISEM 770 GOOGLE/bin/activate"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "✅ Environment variables loaded"
else
    echo "⚠️  No .env file found. Please run setup_google_api.py first."
fi

# Start Jupyter
echo "📊 Starting Jupyter Notebook..."
jupyter notebook
'''
    
    with open('start_project.sh', 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod('start_project.sh', 0o755)
    print("✅ Created startup script: start_project.sh")

def main():
    """Main setup function."""
    print("🎓 ISEM 770 GOOGLE Project Setup")
    print("=" * 40)
    
    # Setup API key
    if not setup_google_api_key():
        sys.exit(1)
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment setup incomplete. Please fix the issues above.")
        sys.exit(1)
    
    # Create startup script
    create_startup_script()
    
    print("\n🎉 Setup Complete!")
    print("=" * 40)
    print("To start working with your project:")
    print("1. Run: ./start_project.sh")
    print("2. Or manually activate: source 'ISEM 770 GOOGLE/bin/activate'")
    print("3. Start Jupyter: jupyter notebook")
    print("\n📚 Available notebooks:")
    print("   - 01_simple_rag.ipynb (Start here!)")
    print("   - 02_semantic_chunking.ipynb")
    print("   - And many more...")

if __name__ == "__main__":
    main() 