#!/usr/bin/env python3
"""
Audio File Splitter for OpenAI Whisper API
Splits large audio files into smaller chunks (under 25MB) for OpenAI processing
"""

import os
import sys
import logging
from pathlib import Path
from typing import List
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioSplitter:
    """Splits large audio files into smaller chunks"""
    
    def __init__(self, input_dir: str = "demo-files-mp3", output_dir: str = "demo-files-split"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_size_mb = 20  # Keep under 25MB limit with some buffer
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Max chunk size: {self.max_size_mb}MB")
    
    def get_audio_duration(self, file_path: Path) -> float:
        """Get audio file duration using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to get duration for {file_path}: {e}")
            return 0.0
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        return file_path.stat().st_size / (1024 * 1024)
    
    def calculate_chunk_duration(self, file_path: Path) -> float:
        """Calculate optimal chunk duration based on file size and duration"""
        file_size_mb = self.get_file_size_mb(file_path)
        duration = self.get_audio_duration(file_path)
        
        if duration == 0:
            return 0.0
        
        # Calculate duration per MB
        duration_per_mb = duration / file_size_mb
        
        # Calculate chunk duration for target size
        chunk_duration = (self.max_size_mb * duration_per_mb) * 0.9  # 90% to be safe
        
        return chunk_duration
    
    def split_audio_file(self, file_path: Path) -> List[Path]:
        """Split a single audio file into chunks"""
        logger.info(f"Splitting {file_path.name}")
        
        duration = self.get_audio_duration(file_path)
        if duration == 0:
            logger.error(f"Could not get duration for {file_path}")
            return []
        
        chunk_duration = self.calculate_chunk_duration(file_path)
        if chunk_duration == 0:
            logger.error(f"Could not calculate chunk duration for {file_path}")
            return []
        
        logger.info(f"File duration: {duration:.1f}s, chunk duration: {chunk_duration:.1f}s")
        
        # Calculate number of chunks needed
        num_chunks = int(duration / chunk_duration) + 1
        
        chunk_files = []
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            if start_time >= duration:
                break
            
            # Create output filename - preserve original format
            base_name = file_path.stem
            original_ext = file_path.suffix
            chunk_name = f"{base_name}_chunk_{i+1:03d}{original_ext}"
            chunk_path = self.output_dir / chunk_name
            
            # Split using ffmpeg
            cmd = [
                'ffmpeg', '-i', str(file_path),
                '-ss', str(start_time),
                '-t', str(chunk_duration),
                '-c', 'copy',  # Copy without re-encoding for speed
                '-y',  # Overwrite output files
                str(chunk_path)
            ]
            
            try:
                logger.info(f"Creating chunk {i+1}/{num_chunks}: {chunk_name}")
                subprocess.run(cmd, capture_output=True, check=True)
                
                # Verify chunk size
                chunk_size = self.get_file_size_mb(chunk_path)
                if chunk_size > 25:
                    logger.warning(f"Chunk {chunk_name} is {chunk_size:.1f}MB - may be too large")
                
                chunk_files.append(chunk_path)
                logger.info(f"Created {chunk_name} ({chunk_size:.1f}MB)")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create chunk {chunk_name}: {e}")
        
        return chunk_files
    
    def split_all_files(self) -> List[Path]:
        """Split all audio files in the input directory"""
        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            return []
        
        # Find all audio files
        audio_files = []
        for ext in ['.mp3', '.m4a', '.wav', '.ogg']:
            audio_files.extend(self.input_dir.glob(f'*{ext}'))
        
        if not audio_files:
            logger.error(f"No audio files found in {self.input_dir}")
            return []
        
        logger.info(f"Found {len(audio_files)} audio files to split")
        
        all_chunks = []
        
        for audio_file in audio_files:
            file_size = self.get_file_size_mb(audio_file)
            logger.info(f"Processing {audio_file.name} ({file_size:.1f}MB)")
            
            if file_size <= self.max_size_mb:
                logger.info(f"File {audio_file.name} is already small enough, copying...")
                # Copy file to output directory
                output_path = self.output_dir / audio_file.name
                import shutil
                shutil.copy2(audio_file, output_path)
                all_chunks.append(output_path)
            else:
                # Split the file
                chunks = self.split_audio_file(audio_file)
                all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} audio chunks in {self.output_dir}")
        return all_chunks


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Split large audio files for OpenAI Whisper API")
    parser.add_argument("--input-dir", default="demo-files-mp3",
                       help="Input directory containing audio files")
    parser.add_argument("--output-dir", default="demo-files-split",
                       help="Output directory for split files")
    parser.add_argument("--max-size", type=float, default=20.0,
                       help="Maximum chunk size in MB (default: 20)")
    
    args = parser.parse_args()
    
    try:
        splitter = AudioSplitter(args.input_dir, args.output_dir)
        splitter.max_size_mb = args.max_size
        
        chunks = splitter.split_all_files()
        
        if chunks:
            logger.info(f"Successfully created {len(chunks)} audio chunks")
            logger.info(f"Chunks saved to: {splitter.output_dir}")
        else:
            logger.error("No chunks were created")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Splitting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()