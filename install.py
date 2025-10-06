#!/usr/bin/env python3
"""
Installation script for Hebrew Audio Word Counter
Automatically installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU processing")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet - CUDA check will be done after installation")
        return False


def install_requirements():
    """Install Python requirements"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                      "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installing Python dependencies"):
        return False
    
    return True


def install_whisperx():
    """Install WhisperX from GitHub"""
    print("🔄 Installing WhisperX from GitHub...")
    
    # Try to install from GitHub
    if run_command(f"{sys.executable} -m pip install git+https://github.com/m-bain/whisperX.git", 
                  "Installing WhisperX"):
        return True
    
    # Fallback to PyPI
    print("🔄 Trying PyPI installation...")
    if run_command(f"{sys.executable} -m pip install whisperx", 
                  "Installing WhisperX from PyPI"):
        return True
    
    print("❌ Failed to install WhisperX")
    return False


def verify_installation():
    """Verify that all components are installed correctly"""
    print("\n🔍 Verifying installation...")
    
    try:
        import whisperx
        print("✅ WhisperX imported successfully")
    except ImportError as e:
        print(f"❌ WhisperX import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    # Check CUDA after installation
    check_cuda()
    
    return True


def create_demo_structure():
    """Create demo directory structure if it doesn't exist"""
    demo_path = Path("demo-files")
    if not demo_path.exists():
        print("📁 Creating demo-files directory structure...")
        demo_path.mkdir()
        
        # Create subdirectories
        (demo_path / "tv").mkdir()
        (demo_path / "radio").mkdir()
        (demo_path / "tech-meetings").mkdir()
        
        print("✅ Demo directory structure created")
        print("   Place your audio files in the appropriate subdirectories:")
        print("   - demo-files/tv/ (for TV audio files)")
        print("   - demo-files/radio/ (for radio audio files)")
        print("   - demo-files/tech-meetings/ (for meeting audio files)")
    else:
        print("✅ Demo directory structure already exists")


def main():
    """Main installation function"""
    print("🚀 Hebrew Audio Word Counter - Installation Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Install WhisperX
    if not install_whisperx():
        print("❌ Failed to install WhisperX")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Create demo structure
    create_demo_structure()
    
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Place your audio files in the demo-files subdirectories")
    print("2. Run the processing script:")
    print("   python process_audio.py")
    print("\n📖 For more information, see README.md")
    
    # Test with a simple command
    print("\n🧪 Testing basic functionality...")
    try:
        result = subprocess.run([sys.executable, "process_audio.py", "--help"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Script is ready to use!")
        else:
            print("⚠️  Script test failed, but installation appears complete")
    except Exception as e:
        print(f"⚠️  Could not test script: {e}")


if __name__ == "__main__":
    main()
