#!/usr/bin/env python3
"""
Convert OGG files to MP3 format for OpenAI Whisper API compatibility
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# Audio processing
try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Installing required audio processing libraries...")
    os.system("pip install librosa soundfile")
    import librosa
    import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ogg_conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def convert_ogg_to_mp3(input_path: Path, output_path: Path) -> bool:
    """Convert a single OGG file to MP3"""
    try:
        logger.info(f"Converting {input_path.name} to MP3...")
        
        # Load OGG file
        audio_data, sample_rate = librosa.load(str(input_path), sr=None)
        duration = len(audio_data) / sample_rate
        
        logger.info(f"Audio info: {duration:.1f}s, {sample_rate}Hz")
        
        # Save as MP3
        sf.write(str(output_path), audio_data, sample_rate, format='MP3', subtype='MPEG_LAYER_III')
        
        # Check output file size
        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Created {output_path.name}: {output_size_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {input_path.name}: {e}")
        return False


def convert_directory(input_dir: Path, output_dir: Path) -> int:
    """Convert all OGG files in a directory to MP3"""
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 0
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all OGG files
    ogg_files = list(input_dir.rglob('*.ogg'))
    
    if not ogg_files:
        logger.warning(f"No OGG files found in {input_dir}")
        return 0
    
    logger.info(f"Found {len(ogg_files)} OGG files to convert")
    
    converted_count = 0
    
    for ogg_file in ogg_files:
        # Calculate relative path to preserve structure
        relative_path = ogg_file.relative_to(input_dir)
        output_subfolder = output_dir / relative_path.parent
        output_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Create MP3 filename
        mp3_file = output_subfolder / f"{ogg_file.stem}.mp3"
        
        if convert_ogg_to_mp3(ogg_file, mp3_file):
            converted_count += 1
    
    return converted_count


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert OGG files to MP3 for OpenAI Whisper API")
    parser.add_argument("--input-dir", default="demo-files",
                       help="Input directory containing OGG files")
    parser.add_argument("--output-dir", default="demo-files-mp3",
                       help="Output directory for MP3 files")
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.input_dir)
        output_path = Path(args.output_dir)
        
        logger.info(f"Converting OGG files from {input_path} to {output_path}")
        converted_count = convert_directory(input_path, output_path)
        
        logger.info(f"\nConversion completed: {converted_count} files converted")
        logger.info(f"MP3 files saved to: {output_path}")
        
        if converted_count > 0:
            logger.info(f"\nYou can now run:")
            logger.info(f"python process_audio_openai.py --demo-files {output_path}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
