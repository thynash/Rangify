@echo off
echo 🕸️ Rangify - SIH Problem Statement ID25107 Solution
echo ============================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if virtual environment exists
if not exist "rangify_env" (
    echo 🔧 Creating virtual environment...
    python -m venv rangify_env
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call rangify_env\Scripts\activate.bat

REM Install requirements if needed
if not exist "rangify_env\Lib\site-packages\streamlit" (
    echo 📦 Installing requirements...
    pip install --upgrade pip
    pip install -r requirements.txt
)

REM Launch Rangify
echo 🚀 Launching Rangify...
python start_kolamai.py

pause