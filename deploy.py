#!/usr/bin/env python3
"""
KolamAI Deployment Script
Simple deployment helper for KolamAI Kolam Pattern Studio
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def check_requirements():
    """Check if all requirements are met"""
    print("📋 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} found")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    print("✅ requirements.txt found")
    
    # Check if streamlit_app.py exists
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found")
        return False
    print("✅ streamlit_app.py found")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def run_local():
    """Run the app locally"""
    print("🚀 Starting KolamAI locally...")
    print("🌐 App will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            "streamlit", "run", "streamlit_app.py",
            "--server.address=0.0.0.0",
            "--server.port=8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 KolamAI stopped")

def create_heroku_files():
    """Create Heroku deployment files"""
    print("📝 Creating Heroku deployment files...")
    
    # Create Procfile
    with open("Procfile", "w") as f:
        f.write("web: sh setup.sh && streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0\n")
    
    # Create setup.sh
    with open("setup.sh", "w") as f:
        f.write("""mkdir -p ~/.streamlit/

echo "\\
[general]\\n\\
email = \\"your-email@domain.com\\"\\n\\
" > ~/.streamlit/credentials.toml

echo "\\
[server]\\n\\
headless = true\\n\\
enableCORS=false\\n\\
port = $PORT\\n\\
" > ~/.streamlit/config.toml
""")
    
    print("✅ Heroku files created: Procfile, setup.sh")

def deploy_heroku(app_name):
    """Deploy to Heroku"""
    print(f"🚀 Deploying to Heroku as '{app_name}'...")
    
    # Check if Heroku CLI is installed
    if not run_command("heroku --version", "Checking Heroku CLI"):
        print("❌ Please install Heroku CLI first: https://devcenter.heroku.com/articles/heroku-cli")
        return False
    
    # Create Heroku files
    create_heroku_files()
    
    # Create Heroku app
    run_command(f"heroku create {app_name}", f"Creating Heroku app '{app_name}'")
    
    # Add and commit files
    run_command("git add .", "Adding files to git")
    run_command('git commit -m "Deploy to Heroku"', "Committing changes")
    
    # Deploy to Heroku
    result = run_command("git push heroku main", "Deploying to Heroku")
    
    if result:
        print(f"🎉 Deployment successful!")
        print(f"🌐 Your app is available at: https://{app_name}.herokuapp.com")
        return True
    
    return False

def show_streamlit_cloud_instructions():
    """Show Streamlit Cloud deployment instructions"""
    print("""
🎯 Streamlit Cloud Deployment Instructions:

1. 📤 Push your code to GitHub:
   git add .
   git commit -m "Ready for Streamlit Cloud"
   git push origin main

2. 🌐 Go to: https://share.streamlit.io

3. 🔗 Connect your GitHub account

4. 📁 Select your repository

5. 📄 Set main file: streamlit_app.py

6. 🚀 Click "Deploy"

✅ Your app will be live in minutes with automatic updates!
""")

def main():
    parser = argparse.ArgumentParser(description="KolamAI Deployment Helper")
    parser.add_argument("action", choices=["local", "heroku", "streamlit-cloud", "setup"], 
                       help="Deployment action")
    parser.add_argument("--app-name", help="App name for Heroku deployment")
    
    args = parser.parse_args()
    
    print("🕸️ KolamAI Deployment Helper")
    print("=" * 40)
    
    # Check requirements for all actions
    if not check_requirements():
        sys.exit(1)
    
    if args.action == "setup":
        print("🔧 Setting up KolamAI...")
        if install_dependencies():
            print("✅ Setup complete! Run 'python deploy.py local' to start")
        else:
            print("❌ Setup failed")
            
    elif args.action == "local":
        print("🏠 Running locally...")
        if install_dependencies():
            run_local()
        else:
            print("❌ Failed to install dependencies")
            
    elif args.action == "heroku":
        if not args.app_name:
            print("❌ Please provide --app-name for Heroku deployment")
            sys.exit(1)
        deploy_heroku(args.app_name)
        
    elif args.action == "streamlit-cloud":
        show_streamlit_cloud_instructions()

if __name__ == "__main__":
    main()