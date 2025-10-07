# OpenAI Whisper API Setup Guide

This guide helps you set up and use the OpenAI Whisper API version of the Hebrew audio processing tool, which is more reliable than the local WhisperX version.

## Why Use OpenAI Whisper API?

- **More Reliable**: No local model downloads or GPU requirements
- **Better Accuracy**: Uses OpenAI's latest Whisper models
- **Easier Setup**: Just need an API key
- **No Hardware Requirements**: Works on any system with internet
- **Automatic Updates**: Always uses the latest model improvements

## Prerequisites

1. **OpenAI Account**: Sign up at [OpenAI Platform](https://platform.openai.com/)
2. **API Key**: Get your API key from [API Keys page](https://platform.openai.com/api-keys)
3. **Python 3.7+**: Make sure Python is installed

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements_openai.txt
```

### 2. Run Setup Script
```bash
python setup_openai.py
```

This will:
- Help you configure your API key
- Test the API connection
- Check your audio files
- Set up environment variables

### 3. Run Processing
```bash
# Windows
run_processing_openai.bat

# Or manually
python process_audio_openai.py
```

## Manual Setup

### 1. Set API Key
**Option A: Environment Variable**
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

**Option B: .env File**
Create a `.env` file in the project directory:
```
OPENAI_API_KEY=your_api_key_here
```

**Option C: Command Line**
```bash
python process_audio_openai.py --api-key your_api_key_here
```

### 2. Test Connection
```python
from openai import OpenAI
client = OpenAI(api_key="your_api_key_here")
models = client.models.list()
print("Connection successful!")
```

## Usage Examples

### Basic Usage
```bash
python process_audio_openai.py
```

### Custom Settings
```bash
# Use specific directory
python process_audio_openai.py --demo-files demo-files-big

# Low CPU mode (with rate limiting)
python process_audio_openai.py --low-cpu

# Custom output files
python process_audio_openai.py --output-json my_results.json --output-csv my_results.csv

# Different language
python process_audio_openai.py --language he
```

### Command Line Options
- `--demo-files`: Path to audio files directory (default: demo-files)
- `--output-json`: JSON output file (default: hebrew_word_analysis_openai.json)
- `--output-csv`: CSV output file (default: hebrew_word_analysis_openai.csv)
- `--api-key`: OpenAI API key (or set OPENAI_API_KEY env var)
- `--model`: Whisper model to use (default: whisper-1)
- `--language`: Language code (default: he for Hebrew)
- `--low-cpu`: Enable rate limiting for lower resource usage

## Supported Audio Formats

- **MP3** (.mp3)
- **MP4** (.mp4)
- **MPEG** (.mpeg)
- **MPGA** (.mpga)
- **M4A** (.m4a)
- **WAV** (.wav)
- **WEBM** (.webm)

## File Size Limits

- **Maximum file size**: 25MB per file
- **No limit on number of files**
- **No limit on total processing time**

## Cost Estimation

OpenAI Whisper API pricing:
- **$0.006 per minute** of audio
- **Example**: 1 hour of audio = $0.36

The script provides cost estimates for each file and total processing.

## Troubleshooting

### Common Issues

**1. API Key Not Found**
```
Error: OpenAI API key not provided
```
**Solution**: Set OPENAI_API_KEY environment variable or use --api-key

**2. File Too Large**
```
Error: File too large: 30.5MB (max 25MB)
```
**Solution**: Split large files or compress them

**3. Unsupported Format**
```
Error: Unsupported format: .flac
```
**Solution**: Convert to supported format (MP3, WAV, etc.)

**4. Rate Limiting**
```
Error: Rate limit exceeded
```
**Solution**: Use --low-cpu flag or wait and retry

### File Conversion

If you have unsupported formats, convert them:

**Using FFmpeg:**
```bash
# Convert FLAC to MP3
ffmpeg -i input.flac -acodec mp3 output.mp3

# Convert to WAV
ffmpeg -i input.m4a -acodec pcm_s16le output.wav
```

**Using Online Tools:**
- [CloudConvert](https://cloudconvert.com/)
- [Online Audio Converter](https://online-audio-converter.com/)

## Performance Tips

### 1. File Organization
- Keep files under 25MB
- Use supported formats
- Organize in subdirectories (tv, radio, tech-meetings)

### 2. Batch Processing
- Process files in smaller batches
- Use --low-cpu for rate limiting
- Monitor API usage

### 3. Cost Optimization
- Compress audio files to reduce size
- Use shorter audio clips when possible
- Monitor cost estimates in logs

## Output Files

### JSON Output (`hebrew_word_analysis_openai.json`)
```json
{
  "radio": {
    "subfolder": "radio",
    "files_processed": 5,
    "files_failed": 0,
    "total_hebrew_words": 1250,
    "unique_hebrew_words": 450,
    "top_1000_words": [["מילה", 25], ["עוד", 20], ...],
    "estimated_cost": 0.15
  }
}
```

### CSV Output (`hebrew_word_analysis_openai.csv`)
```csv
subfolder,rank,word,count,total_files,total_words,estimated_cost
radio,1,מילה,25,5,1250,0.15
radio,2,עוד,20,5,1250,0.15
```

## Comparison: Local vs OpenAI

| Feature | Local WhisperX | OpenAI Whisper |
|---------|----------------|----------------|
| Setup Complexity | High | Low |
| Hardware Requirements | GPU recommended | None |
| Model Updates | Manual | Automatic |
| Reliability | Variable | High |
| Cost | Free (after setup) | $0.006/minute |
| File Size Limit | None | 25MB |
| Internet Required | No | Yes |

## Security Notes

- **Never commit API keys** to version control
- **Use environment variables** for API keys
- **Monitor API usage** to avoid unexpected charges
- **Set usage limits** in OpenAI dashboard

## Getting Help

1. **Check logs**: `audio_processing_openai.log`
2. **Test API**: Run `python setup_openai.py`
3. **Verify files**: Check file formats and sizes
4. **OpenAI Support**: [OpenAI Help Center](https://help.openai.com/)

## Next Steps

1. Run the setup script: `python setup_openai.py`
2. Test with a small audio file
3. Process your full dataset
4. Analyze results in the generated CSV/JSON files
