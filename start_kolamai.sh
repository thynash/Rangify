#!/bin/bash

echo "🕸️ KolamAI - SIH Problem Statement ID25107 Solution"
echo "============================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found"

# Check if virtual environment exists
if [ ! -d "kolamai_env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv kolamai_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source kolamai_env/bin/activate

# Install requirements if needed
if [ ! -f "kolamai_env/lib/python*/site-packages/streamlit" ]; then
    echo "📦 Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Launch KolamAI
echo "🚀 Launching KolamAI..."
python3 start_kolamai.py