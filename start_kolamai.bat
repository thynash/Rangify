@echo off
echo 🕸️ KolamAI - SIH Problem Statement ID25107 Solution
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
if not exist "kolamai_env" (
    echo 🔧 Creating virtual environment...
    python -m venv kolamai_env
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call kolamai_env\Scripts\activate.bat

REM Install requirements if needed
if not exist "kolamai_env\Lib\site-packages\streamlit" (
    echo 📦 Installing requirements...
    pip install --upgrade pip
    pip install -r requirements.txt
)

REM Launch KolamAI
echo 🚀 Launching KolamAI...
python start_kolamai.py

pause