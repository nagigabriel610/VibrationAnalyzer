"""
Simple test to verify your setup is working
Run this first to check if everything is installed correctly
"""

import os
import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import sys
        print("✅ sys - OK")
        
        import os
        print("✅ os - OK")
        
        import numpy as np
        print("✅ numpy - OK")
        
        import pandas as pd
        print("✅ pandas - OK")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib - OK")
        
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5 - OK")
        
        import scipy.signal
        print("✅ scipy - OK")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_file_structure():
    """Test if files are in the right places"""
    print("\nTesting file structure...")
    
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "src/gui/main_window.py",
        "src/analysis/signal_processor.py",
        "src/data/mock_generator.py"
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - Found")
        else:
            print(f"❌ {file_path} - Missing")
            all_good = False
    
    return all_good

def create_missing_init_files():
    """Create missing __init__.py files"""
    print("\nCreating __init__.py files...")
    
    init_files = [
        "src/__init__.py",
        "src/gui/__init__.py", 
        "src/analysis/__init__.py",
        "src/data/__init__.py"
    ]
    
    for init_file in init_files:
        dir_path = os.path.dirname(init_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
            
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Package initialization file\n")
            print(f"Created: {init_file}")
        else:
            print(f"✅ {init_file} already exists")

if __name__ == "__main__":
    print("=" * 50)
    print("VIBRATION ANALYZER SETUP TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n⚠️  Please install missing packages first:")
        print("pip install -r requirements.txt")
        input("Press Enter to exit...")
        exit()
    
    # Create missing files
    create_missing_init_files()
    
    # Test file structure
    files_ok = test_file_structure()
    
    if files_ok:
        print("\n🎉 Setup looks good! Try running main.py now")
    else:
        print("\n⚠️  Some files are missing. Please check the file structure.")
    
    input("Press Enter to continue...")