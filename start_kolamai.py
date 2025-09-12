#!/usr/bin/env python3
"""
KolamAI Startup Script
Easy launcher for SIH demonstration
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import streamlit
        import cv2
        import numpy
        import plotly
        import pandas
        print("✅ All required packages found")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_data_directory():
    """Check if data directory exists"""
    data_dir = Path("data")
    if data_dir.exists():
        image_files = list(data_dir.glob("*.jpg"))
        print(f"✅ Found {len(image_files)} sample images in data directory")
        return True
    else:
        print("⚠️ Data directory not found. Some features may not work.")
        return False

def main():
    """Main startup function"""
    
    print("🕸️ KolamAI - SIH Problem Statement ID25107 Solution")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check data
    check_data_directory()
    
    # Check if modules exist
    required_files = ["streamlit_app.py", "kolam_analyzer.py", "kolam_generator.py", "main.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        sys.exit(1)
    
    print("✅ All core modules found")
    
    # Launch options
    print("\n🚀 Launch Options:")
    print("1. Web Application (Recommended for SIH Demo)")
    print("2. Command Line Interface")
    print("3. Complete Demo")
    print("4. Quick Analysis Test")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            launch_web_app()
        elif choice == "2":
            launch_cli()
        elif choice == "3":
            run_demo()
        elif choice == "4":
            run_quick_test()
        else:
            print("Invalid choice. Launching web app by default...")
            launch_web_app()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

def launch_web_app():
    """Launch the Streamlit web application"""
    
    print("\n🌐 Launching KolamAI Web Application...")
    print("📍 URL: http://localhost:8501")
    print("⚠️ Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Try using main.py first
        subprocess.run([sys.executable, "main.py", "web", "--port", "8501"])
    except FileNotFoundError:
        # Fallback to direct streamlit command
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"])

def launch_cli():
    """Launch command line interface"""
    
    print("\n💻 KolamAI Command Line Interface")
    print("Available commands:")
    print("  python main.py analyze --input <image> --verbose")
    print("  python main.py generate --symmetry radial --style tamil")
    print("  python main.py demo --input data --output results")
    print("\nFor full help: python main.py --help")

def run_demo():
    """Run complete demonstration"""
    
    print("\n🎪 Running Complete KolamAI Demo...")
    
    try:
        subprocess.run([sys.executable, "main.py", "demo", "--input", "data", "--output", "demo_results"])
        print("\n✅ Demo completed! Check demo_results/ directory for outputs.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def run_quick_test():
    """Run quick functionality test"""
    
    print("\n🧪 Running Quick Functionality Test...")
    
    # Test imports
    try:
        from kolam_analyzer import KolamAnalyzer
        from kolam_generator import KolamGenerator, GenerationParams, SymmetryType
        print("✅ Core modules import successfully")
        
        # Test generator
        generator = KolamGenerator()
        params = GenerationParams(
            canvas_size=(256, 256),
            symmetry_type=SymmetryType.RADIAL,
            complexity_level=0.5
        )
        
        print("🎨 Testing pattern generation...")
        kolam = generator.generate_kolam(params)
        print("✅ Pattern generation works")
        
        # Test analyzer if sample image exists
        data_dir = Path("data")
        if data_dir.exists():
            sample_images = list(data_dir.glob("*.jpg"))
            if sample_images:
                print("🔍 Testing pattern analysis...")
                analyzer = KolamAnalyzer()
                pattern = analyzer.analyze_image(str(sample_images[0]))
                print(f"✅ Analysis works - detected {pattern.symmetry_type} symmetry")
        
        print("\n🎉 All tests passed! KolamAI is ready for SIH demo.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()