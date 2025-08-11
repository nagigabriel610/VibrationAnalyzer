"""
Vibration Analyzer Pro - Main Application Entry Point
Professional vibration analysis software for industrial applications

Author: Your Development Team
Version: 1.0.0
"""

import sys
import os

def main():
    """Main application entry point with proper path handling"""
    
    print("Starting Vibration Analyzer Pro...")
    print("=" * 50)
    print("Professional Vibration Signal Analysis Software")
    print("Version 1.0.0")
    print("=" * 50)
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    
    # Add src directory to Python path if not already there
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Debug: Print paths to help troubleshooting
    print(f"Current directory: {current_dir}")
    print(f"Source directory: {src_dir}")
    print(f"Python path includes: {sys.path[:3]}...")  # First 3 entries
    
    # Check if required files exist
    gui_file = os.path.join(src_dir, 'gui', 'main_window.py')
    if not os.path.exists(gui_file):
        print(f"ERROR: GUI file not found at: {gui_file}")
        print("Please make sure all files are in the correct locations.")
        input("Press Enter to exit...")
        return
    
    try:
        # Import and run main GUI
        from gui.main_window import main as gui_main
        gui_main()
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're running from the VibrationAnalyzer directory")
        print("2. Make sure all __init__.py files exist")
        print("3. Make sure the virtual environment is activated")
        print("4. Try: pip install -r requirements.txt")
        input("Press Enter to exit...")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()