# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
python install.py
```

### 2. Place Your Audio Files
Put your audio files in the appropriate subdirectories:
- `demo-files/tv/` - for TV audio files
- `demo-files/radio/` - for radio audio files  
- `demo-files/tech-meetings/` - for meeting audio files

**Supported formats:** MP3, OGG, M4A, WAV, FLAC

### 3. Run Processing
```bash
# Windows (double-click or run in command prompt)
run_processing.bat

# Or manually:
python process_audio.py
```

## 📊 Results

After processing, you'll get:
- `hebrew_word_analysis.json` - Complete analysis
- `hebrew_word_analysis.csv` - Spreadsheet format
- `audio_processing.log` - Detailed logs

## ⚡ Quick Test

Test the setup without processing audio:
```bash
python test_structure.py
```

## 🔧 Troubleshooting

**No GPU?** The script will automatically use CPU (slower but works)

**Out of memory?** Use smaller batch size:
```bash
python process_audio.py --batch-size 4 --device cpu
```

**Need help?** Check `README.md` for detailed documentation
