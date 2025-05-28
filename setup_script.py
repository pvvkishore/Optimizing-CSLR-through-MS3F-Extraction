#!/usr/bin/env python3
"""
Setup script for Sign Language Model Inference
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages"""
    packages = [
        'torch',
        'torchvision', 
        'opencv-python',
        'matplotlib',
        'seaborn',
        'pandas',
        'numpy',
        'tqdm',
        'scikit-learn',
        'editdistance',
        'pillow'
    ]
    
    print("🔧 Installing required packages...")
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"   ✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            print(f"   You may need to install it manually: pip install {package}")

def install_transformers():
    """Install transformers (optional)"""
    try:
        print("🤖 Installing transformers (for text encoder)...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers'])
        print("   ✅ Transformers installed")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Transformers installation failed: {e}")
        print("   Text encoder will be disabled, but inference will still work")

def check_paths():
    """Check if required files exist"""
    paths = {
        'Model': '/home/pvvkishore/Desktop/TVC_May21/New_Code/checkpoints_112x112/final_model_112x112.pth',
        'Test Data': '/home/pvvkishore/Desktop/TVC_May21/New_Code/Test/',
        'Annotations': '/home/pvvkishore/Desktop/TVC_May21/New_Code/annotations_folder/test_gloss_eng.csv'
    }
    
    print("📁 Checking required files...")
    all_exist = True
    
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def main():
    print("🚀 SIGN LANGUAGE MODEL INFERENCE - SETUP")
    print("=" * 50)
    
    # Install packages
    install_packages()
    print()
    
    # Install transformers (optional)
    install_transformers()
    print()
    
    # Check paths
    if check_paths():
        print("\n✅ Setup completed successfully!")
        print("📋 Next steps:")
        print("   1. Save the inference script as 'inference.py'")
        print("   2. Run: python inference.py")
        print("   3. Check the 'results' folder for outputs")
    else:
        print("\n⚠️  Some files are missing. Please check the paths above.")
        print("   You may need to adjust the paths in the inference script.")

if __name__ == "__main__":
    main()
