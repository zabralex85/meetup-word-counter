#!/usr/bin/env python3
"""
Setup script for OpenAI Whisper API integration
Helps configure API key and test the connection
"""

import os
import sys
from pathlib import Path

def setup_api_key():
    """Setup OpenAI API key"""
    print("=" * 50)
    print("OpenAI Whisper API Setup")
    print("=" * 50)
    print()
    
    # Check if API key is already set
    existing_key = os.getenv('OPENAI_API_KEY')
    if existing_key:
        print(f"✓ OpenAI API key is already set: {existing_key[:8]}...")
        choice = input("Do you want to update it? (y/n): ").lower().strip()
        if choice != 'y':
            return existing_key
    
    print("Please enter your OpenAI API key:")
    print("(You can get one from: https://platform.openai.com/api-keys)")
    print()
    
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided!")
        return None
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key should start with 'sk-'")
        confirm = input("Continue anyway? (y/n): ").lower().strip()
        if confirm != 'y':
            return None
    
    # Set environment variable for current session
    os.environ['OPENAI_API_KEY'] = api_key
    
    # Try to save to .env file
    env_file = Path('.env')
    try:
        with open(env_file, 'w') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        print(f"✓ API key saved to {env_file}")
    except Exception as e:
        print(f"⚠️  Could not save to .env file: {e}")
        print("You'll need to set the environment variable manually for each session.")
    
    return api_key

def test_connection(api_key):
    """Test OpenAI API connection"""
    print("\n" + "=" * 30)
    print("Testing API Connection...")
    print("=" * 30)
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test with a simple API call
        models = client.models.list()
        whisper_available = any(model.id == 'whisper-1' for model in models.data)
        
        if whisper_available:
            print("✓ OpenAI API connection successful!")
            print("✓ Whisper model is available")
            return True
        else:
            print("❌ Whisper model not available")
            return False
            
    except ImportError:
        print("❌ OpenAI library not installed")
        print("Run: pip install openai")
        return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def check_audio_files():
    """Check for audio files in demo directories"""
    print("\n" + "=" * 30)
    print("Checking Audio Files...")
    print("=" * 30)
    
    demo_path = Path("demo-files")
    if not demo_path.exists():
        print("❌ demo-files directory not found")
        return False
    
    supported_formats = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
    total_files = 0
    large_files = 0
    
    for subdir in demo_path.iterdir():
        if subdir.is_dir():
            files_in_dir = 0
            for file_path in subdir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                    files_in_dir += 1
                    total_files += 1
                    
                    # Check file size
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb > 25:
                        large_files += 1
                        print(f"⚠️  Large file: {file_path.name} ({size_mb:.1f}MB)")
            
            if files_in_dir > 0:
                print(f"✓ {subdir.name}: {files_in_dir} audio files")
    
    if total_files == 0:
        print("❌ No supported audio files found")
        print("Supported formats: MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM")
        return False
    
    print(f"✓ Total: {total_files} audio files found")
    if large_files > 0:
        print(f"⚠️  {large_files} files exceed 25MB limit")
    
    return True

def main():
    """Main setup function"""
    print("Setting up OpenAI Whisper integration for Hebrew audio processing...")
    print()
    
    # Setup API key
    api_key = setup_api_key()
    if not api_key:
        print("\n❌ Setup failed: No valid API key provided")
        sys.exit(1)
    
    # Test connection
    if not test_connection(api_key):
        print("\n❌ Setup failed: API connection test failed")
        sys.exit(1)
    
    # Check audio files
    if not check_audio_files():
        print("\n⚠️  Setup completed but no audio files found")
        print("Add audio files to demo-files subdirectories and run again")
    else:
        print("\n✓ Setup completed successfully!")
        print("\nYou can now run:")
        print("  run_processing_openai.bat")
        print("  or")
        print("  python process_audio_openai.py")
    
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
