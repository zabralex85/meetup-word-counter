#!/usr/bin/env python3
"""
Audio File Splitter for OpenAI Whisper API
Splits large audio files into smaller chunks that fit within the 25MB limit
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# Audio processing
try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Installing required audio processing libraries...")
    os.system("pip install librosa soundfile")
    import librosa
    import soundfile as sf
    import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_splitting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AudioSplitter:
    """Splits large audio files into smaller chunks"""
    
    def __init__(self, max_size_mb: float = 20.0, chunk_duration_minutes: float = 20.0):
        self.max_size_mb = max_size_mb
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        
        logger.info(f"Audio splitter initialized: max {max_size_mb}MB, {chunk_duration_minutes}min chunks")
    
    def estimate_file_size(self, duration_seconds: float, sample_rate: int = 44100, 
                          channels: int = 2, bit_depth: int = 16) -> float:
        """Estimate file size in MB"""
        # Rough estimation: duration * sample_rate * channels * (bit_depth/8) / (1024*1024)
        size_bytes = duration_seconds * sample_rate * channels * (bit_depth / 8)
        return size_bytes / (1024 * 1024)
    
    def split_audio_file(self, input_path: Path, output_dir: Path) -> List[Path]:
        """Split a single audio file into smaller chunks"""
        logger.info(f"Splitting {input_path.name}")
        
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(str(input_path), sr=None)
            duration_seconds = len(audio_data) / sample_rate
            
            logger.info(f"Audio info: {duration_seconds:.1f}s, {sample_rate}Hz, {len(audio_data)} samples")
            
            # Calculate chunk size in samples
            chunk_samples = int(self.chunk_duration_seconds * sample_rate)
            total_chunks = int(np.ceil(len(audio_data) / chunk_samples))
            
            logger.info(f"Will create {total_chunks} chunks of ~{self.chunk_duration_seconds/60:.1f} minutes each")
            
            output_files = []
            
            for i in range(total_chunks):
                start_sample = i * chunk_samples
                end_sample = min((i + 1) * chunk_samples, len(audio_data))
                chunk_data = audio_data[start_sample:end_sample]
                
                # Create output filename
                chunk_name = f"{input_path.stem}_chunk_{i+1:03d}.mp3"
                chunk_path = output_dir / chunk_name
                
                # Save chunk as MP3 with optimized compression (smaller file size)
                sf.write(str(chunk_path), chunk_data, sample_rate, format='MP3', subtype='MPEG_LAYER_III')
                
                # Verify file size
                chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
                logger.info(f"Created {chunk_name}: {chunk_size_mb:.1f}MB")
                
                if chunk_size_mb > self.max_size_mb:
                    logger.warning(f"Chunk {chunk_name} is {chunk_size_mb:.1f}MB (exceeds {self.max_size_mb}MB limit)")
                
                output_files.append(chunk_path)
            
            logger.info(f"Successfully split {input_path.name} into {len(output_files)} chunks")
            return output_files
            
        except Exception as e:
            logger.error(f"Failed to split {input_path.name}: {e}")
            return []
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, List[Path]]:
        """Process all audio files in a directory while preserving folder structure"""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                audio_files.append(file_path)
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return {}
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        results = {}
        
        for audio_file in audio_files:
            try:
                # Check if file needs splitting
                file_size_mb = audio_file.stat().st_size / (1024 * 1024)
                
                # Preserve folder structure: calculate relative path
                relative_path = audio_file.relative_to(input_dir)
                subfolder = relative_path.parent
                
                # Create corresponding output subfolder
                output_subfolder = output_dir / subfolder
                output_subfolder.mkdir(parents=True, exist_ok=True)
                
                if file_size_mb <= self.max_size_mb:
                    logger.info(f"{audio_file.name} is {file_size_mb:.1f}MB - no splitting needed")
                    # Copy to output directory preserving structure
                    output_file = output_subfolder / audio_file.name
                    import shutil
                    shutil.copy2(audio_file, output_file)
                    results[str(relative_path)] = [output_file]
                else:
                    logger.info(f"{audio_file.name} is {file_size_mb:.1f}MB - needs splitting")
                    # Split the file in the appropriate subfolder
                    chunk_files = self.split_audio_file(audio_file, output_subfolder)
                    results[str(relative_path)] = chunk_files
                    
            except Exception as e:
                logger.error(f"Failed to process {audio_file.name}: {e}")
                results[str(relative_path)] = []
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Split large audio files for OpenAI Whisper API")
    parser.add_argument("--input-dir", default="demo-files-big",
                       help="Input directory containing audio files")
    parser.add_argument("--output-dir", default="demo-files-split",
                       help="Output directory for split files")
    parser.add_argument("--max-size", type=float, default=20.0,
                       help="Maximum file size in MB (default: 20)")
    parser.add_argument("--chunk-duration", type=float, default=20.0,
                       help="Chunk duration in minutes (default: 20)")
    
    args = parser.parse_args()
    
    try:
        # Initialize splitter
        splitter = AudioSplitter(
            max_size_mb=args.max_size,
            chunk_duration_minutes=args.chunk_duration
        )
        
        # Process directory
        input_path = Path(args.input_dir)
        output_path = Path(args.output_dir)
        
        logger.info(f"Splitting audio files from {input_path} to {output_path}")
        results = splitter.process_directory(input_path, output_path)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("SPLITTING SUMMARY")
        logger.info("="*50)
        
        total_original = 0
        total_chunks = 0
        
        for filename, chunk_files in results.items():
            if chunk_files:
                total_original += 1
                total_chunks += len(chunk_files)
                logger.info(f"{filename}: {len(chunk_files)} chunks created")
            else:
                logger.info(f"{filename}: FAILED")
        
        logger.info(f"\nTotal: {total_original} files processed, {total_chunks} chunks created")
        logger.info(f"Output directory: {output_path}")
        
    except Exception as e:
        logger.error(f"Splitting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
