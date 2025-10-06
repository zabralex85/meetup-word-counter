# Hebrew Audio Word Counter

A Python script that processes Hebrew audio files using WhisperX and calculates the top 1000 most frequent Hebrew words for each subfolder.

## Features

- **Multi-format Support**: Processes MP3, OGG, M4A, WAV, and FLAC audio files
- **Hebrew Text Processing**: Specialized Hebrew word extraction and normalization
- **WhisperX Integration**: Uses state-of-the-art speech recognition with word-level timestamps
- **Batch Processing**: Processes all subfolders automatically
- **Comprehensive Output**: Generates both JSON and CSV reports
- **Error Handling**: Robust error handling with detailed logging

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install WhisperX** (if not already installed):
   ```bash
   pip install git+https://github.com/m-bain/whisperX.git
   ```

4. **For Hebrew alignment models** (optional but recommended):
   ```bash
   # You may need to accept the license for wav2vec2 models
   # Visit: https://huggingface.co/microsoft/wav2vec2-large-xlsr-53
   ```

## Usage

### Basic Usage

Process all audio files in the `demo-files` directory:

```bash
python process_audio.py
```

### Advanced Usage

```bash
python process_audio.py \
    --demo-files demo-files \
    --output-json results.json \
    --output-csv results.csv \
    --model-size large-v2 \
    --batch-size 16 \
    --device cuda
```

### Command Line Options

- `--demo-files`: Path to directory containing audio subfolders (default: "demo-files")
- `--output-json`: Output JSON file path (default: "hebrew_word_analysis.json")
- `--output-csv`: Output CSV file path (default: "hebrew_word_analysis.csv")
- `--model-size`: WhisperX model size (tiny, base, small, medium, large, large-v2, large-v3)
- `--batch-size`: Batch size for processing (default: 16)
- `--device`: Device to use (auto, cuda, cpu)

## Directory Structure

The script expects the following structure:

```
demo-files/
├── tv/
│   ├── file1.m4a
│   └── file2.m4a
├── radio/
│   ├── file1.mp3
│   └── file2.mp3
└── tech-meetings/
    ├── file1.ogg
    └── file2.ogg
```

## Output Format

### JSON Output

```json
{
  "tv": {
    "subfolder": "tv",
    "files_processed": 2,
    "files_failed": 0,
    "total_hebrew_words": 1500,
    "unique_hebrew_words": 800,
    "top_1000_words": [
      ["מילה", 45],
      ["עוד", 32],
      ...
    ]
  }
}
```

### CSV Output

| subfolder | rank | word | count | total_files | total_words |
|-----------|------|------|-------|-------------|-------------|
| tv | 1 | מילה | 45 | 2 | 1500 |
| tv | 2 | עוד | 32 | 2 | 1500 |

## Hebrew Text Processing

The script includes specialized Hebrew text processing:

- **Hebrew Character Detection**: Uses Unicode ranges (U+0590-U+05FF)
- **Word Normalization**: Removes punctuation, preserves Hebrew characters
- **Stop Word Filtering**: Filters common Hebrew stop words
- **Diacritics Handling**: Preserves Hebrew diacritics by default

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 4GB GPU memory (for GPU processing)

### Recommended Requirements
- Python 3.9+
- 16GB RAM
- 8GB+ GPU memory (RTX 3070 or better)
- CUDA 11.8+

## Performance Notes

- **GPU Processing**: Significantly faster than CPU (10-50x speedup)
- **Model Size**: Larger models (large-v2, large-v3) provide better accuracy but require more memory
- **Batch Size**: Adjust based on available GPU memory
- **Memory Management**: Script automatically cleans up GPU memory between batches

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   python process_audio.py --device cpu --batch-size 4
   ```

2. **Model Download Issues**:
   - Ensure stable internet connection
   - Check Hugging Face model access permissions

3. **Hebrew Alignment Model Issues**:
   - Script will continue without alignment (less accurate timestamps)
   - Check Hugging Face token permissions

### Logs

Check `audio_processing.log` for detailed processing logs and error information.

## Example Results

After processing, you'll get:

- **Processing Summary**: Files processed, total words, top words per subfolder
- **JSON Report**: Complete analysis with word frequencies
- **CSV Report**: Tabular data for further analysis
- **Log File**: Detailed processing information

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project uses WhisperX which is licensed under BSD-2-Clause license.
