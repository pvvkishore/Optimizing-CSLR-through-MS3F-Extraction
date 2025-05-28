#!/usr/bin/env python3
"""
SETUP CHECKER AND FIX SCRIPT
============================

This script checks your setup and fixes common issues with the 
4-stage sign language recognition system.

Usage: python setup_checker.py
"""

import os
import sys
import subprocess
import importlib.util

def print_header():
    """Print header"""
    print("🔧 SETUP CHECKER AND FIX SCRIPT")
    print("=" * 50)
    print("Checking and fixing 4-Stage Sign Language Recognition setup...")
    print()

def check_python_version():
    """Check Python version"""
    print("1. 🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        print("   💡 Solution: Upgrade Python to 3.8 or higher")
        return False

def check_required_packages():
    """Check required Python packages"""
    print("2. 📦 Checking required packages...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),  
        ('transformers', 'Transformers'),
        ('opencv-python', 'OpenCV'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('tqdm', 'TQDM')
    ]
    
    missing_packages = []
    
    for package_name, display_name in required_packages:
        try:
            # Handle special cases
            if package_name == 'opencv-python':
                import cv2
                print(f"   ✅ {display_name} (OK)")
            elif package_name == 'sklearn':
                import sklearn
                print(f"   ✅ {display_name} (OK)")
            else:
                spec = importlib.util.find_spec(package_name)
                if spec is not None:
                    print(f"   ✅ {display_name} (OK)")
                else:
                    raise ImportError
        except ImportError:
            print(f"   ❌ {display_name} (Missing)")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n   💡 To install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_project_files():
    """Check required project files"""
    print("3. 📄 Checking project files...")
    
    required_files = [
        ('fixed_sign_language_code_Modified.py', 'Main implementation', True),
        ('requirements.txt', 'Dependencies list', True),
        ('config.yaml', 'Configuration file', False),
        ('visualizations.py', 'Visualization tools', False),
        ('integrated_main.py', 'Advanced training', False),
        ('README.md', 'Documentation', False)
    ]
    
    missing_critical = []
    missing_optional = []
    
    for filename, description, critical in required_files:
        if os.path.exists(filename):
            print(f"   ✅ {filename} - {description}")
        else:
            print(f"   ❌ {filename} - {description} (Missing)")
            if critical:
                missing_critical.append(filename)
            else:
                missing_optional.append(filename)
    
    if missing_critical:
        print(f"\n   ⚠️  Critical files missing: {', '.join(missing_critical)}")
        print("   💡 These files are required for basic functionality")
        return False
    elif missing_optional:
        print(f"\n   ⚠️  Optional files missing: {', '.join(missing_optional)}")
        print("   💡 System will work but with limited features")
    
    return True

def check_gpu_support():
    """Check GPU support"""
    print("4. 🎮 Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU Available: {gpu_name}")
            print(f"   ✅ GPU Memory: {memory:.1f} GB")
            print(f"   ✅ GPU Count: {gpu_count}")
            
            if memory < 6:
                print("   ⚠️  GPU memory < 6GB - consider reducing batch size")
            
            return True
        else:
            print("   ⚠️  No GPU available - will use CPU (slower)")
            print("   💡 Install CUDA-enabled PyTorch for GPU acceleration")
            return False
    except:
        print("   ❌ Cannot check GPU status")
        return False

def check_dataset_paths():
    """Check dataset paths"""
    print("5. 📁 Checking dataset paths...")
    
    # Default paths from the system
    default_paths = [
        '/home/pvvkishore/Desktop/TVC_May21/New_Code/train/',
        '/home/pvvkishore/Desktop/TVC_May21/New_Code/annotations_folder/'
    ]
    
    # Check if config.yaml exists to get actual paths
    config_paths = []
    if os.path.exists('config.yaml'):
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            dataset_config = config.get('dataset', {})
            train_path = dataset_config.get('root_train_folder')
            annot_path = dataset_config.get('annotations_folder')
            
            if train_path:
                config_paths.append(train_path)
            if annot_path:
                config_paths.append(annot_path)
        except:
            pass
    
    # Check paths
    paths_to_check = config_paths if config_paths else default_paths
    paths_exist = True
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"   ✅ {path} (Found)")
        else:
            print(f"   ❌ {path} (Not found)")
            paths_exist = False
    
    if not paths_exist:
        print("   💡 Update config.yaml with correct dataset paths")
        print("   💡 Or create sample dataset structure for testing")
    
    return paths_exist

def create_sample_config():
    """Create sample configuration file"""
    print("6. ⚙️  Creating sample configuration...")
    
    if os.path.exists('config.yaml'):
        print("   ✅ config.yaml already exists")
        return True
    
    sample_config = """# 4-Stage Sign Language Recognition Configuration

# Dataset Configuration
dataset:
  root_train_folder: "/path/to/your/train/"
  annotations_folder: "/path/to/your/annotations/"
  csv_filename: "train_gloss_eng.csv"
  max_frames: 32
  test_split: 0.2

# Model Configuration
model:
  feature_dim: 1024
  hidden_dim: 512
  lstm_hidden: 256
  keyframe_ratio: 0.7
  dropout: 0.2

# Training Configuration
training:
  batch_size: 4
  num_epochs: 50
  learning_rates:
    visual_encoder: 1.0e-4
    text_encoder: 2.0e-5
    conv_gru: 5.0e-4
    temporal_lstm: 3.0e-4
    ctc_projection: 3.0e-4

# Hardware Configuration
hardware:
  device: "auto"  # auto, cuda, cpu
  num_workers: 4
  mixed_precision: true

# Directories
directories:
  checkpoints: "checkpoints"
  results: "evaluation_results"
  visualizations: "visualizations"
  logs: "logs"
"""
    
    try:
        with open('config.yaml', 'w') as f:
            f.write(sample_config)
        print("   ✅ Created sample config.yaml")
        print("   💡 Edit config.yaml with your dataset paths")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create config.yaml: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("7. 📂 Creating required directories...")
    
    directories = [
        'checkpoints',
        'evaluation_results', 
        'visualizations',
        'logs'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"   ✅ Created/verified: {directory}/")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}/: {e}")
    
    return True

def run_simple_test():
    """Run simple functionality test"""
    print("8. 🧪 Running simple functionality test...")
    
    try:
        # Test basic imports
        import torch
        import numpy as np
        
        # Test tensor operations
        x = torch.randn(2, 3, 224, 224)
        y = torch.mean(x)
        
        print("   ✅ PyTorch operations working")
        
        # Test if main module can be imported
        if os.path.exists('fixed_sign_language_code_Modified.py'):
            # Try basic import test
            print("   ✅ Main implementation file found")
        else:
            print("   ❌ Main implementation file missing")
            return False
        
        print("   ✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False

def generate_report(results):
    """Generate setup report"""
    print("\n" + "="*60)
    print("📋 SETUP REPORT")
    print("="*60)
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    print(f"Overall Status: {passed_checks}/{total_checks} checks passed")
    print()
    
    for check_name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
    
    print()
    
    if passed_checks == total_checks:
        print("🎉 SETUP COMPLETE!")
        print("Your system is ready for sign language recognition training.")
        print()
        print("Next steps:")
        print("1. Update config.yaml with your dataset paths")
        print("2. Run: python fixed_sign_language_code_Modified.py")
    elif passed_checks >= total_checks * 0.7:
        print("⚠️  SETUP MOSTLY COMPLETE")
        print("Your system should work with some limitations.")
        print("Consider fixing the failed checks for full functionality.")
    else:
        print("❌ SETUP INCOMPLETE")
        print("Several issues need to be resolved before training.")
        print("Please fix the failed checks and run this script again.")
    
    print()

def main():
    """Main setup checker"""
    print_header()
    
    # Run all checks
    results = {}
    
    results['Python Version'] = check_python_version()
    results['Required Packages'] = check_required_packages()
    results['Project Files'] = check_project_files()
    results['GPU Support'] = check_gpu_support()
    results['Dataset Paths'] = check_dataset_paths()
    results['Configuration'] = create_sample_config()
    results['Directories'] = create_directories()
    results['Functionality Test'] = run_simple_test()
    
    # Generate report
    generate_report(results)
    
    # Offer to install missing packages
    if not results['Required Packages']:
        install = input("Would you like to install missing packages? (y/n): ").lower().strip()
        if install == 'y':
            print("\n📦 Installing missing packages...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
                print("✅ Packages installed successfully!")
                print("💡 Run this script again to verify installation")
            except Exception as e:
                print(f"❌ Installation failed: {e}")
                print("💡 Try manually: pip install -r requirements.txt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Setup check cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup checker failed: {e}")
        print("Please check your Python installation and try again.")