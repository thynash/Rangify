#!/bin/bash

echo "🕸️ Rangify - SIH Problem Statement ID25107 Solution"
echo "============================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found"

# Check if virtual environment exists
if [ ! -d "rangify_env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv rangify_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source rangify_env/bin/activate

# Install requirements if needed
if [ ! -f "rangify_env/lib/python*/site-packages/streamlit" ]; then
    echo "📦 Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Launch Rangify
echo "🚀 Launching Rangify - Kolam Pattern Studio..."
echo "🌐 Local URL: http://localhost:8501"
echo "🌐 Network URL: http://0.0.0.0:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo "============================================================"

# Run Streamlit with network access
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501