#!/usr/bin/env python3
"""
Hebrew Audio Processing Script with OpenAI Whisper API
Processes audio files in demo-files subdirectories and calculates top 1000 Hebrew words
Uses OpenAI's Whisper API for reliable transcription
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import re
import time
import base64

# OpenAI imports
import openai
from openai import OpenAI

# Data processing
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_processing_openai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HebrewWordProcessor:
    """Processes Hebrew text and extracts word statistics"""
    
    def __init__(self, low_cpu_mode: bool = False):
        # Hebrew Unicode ranges
        self.hebrew_pattern = re.compile(r'[\u0590-\u05FF\u200F\u200E]+')
        self.low_cpu_mode = low_cpu_mode
        
        # Common Hebrew stop words (can be expanded)
        self.stop_words = {
            'את', 'של', 'על', 'אל', 'עם', 'ב', 'ל', 'ה', 'ו', 'כ', 'מ', 'א', 'י', 'ר', 'ת', 'נ',
            'הוא', 'היא', 'אני', 'אתה', 'אתם', 'אתן', 'אנחנו', 'הם', 'הן', 'זה', 'זאת', 'אלה',
            'אבל', 'או', 'גם', 'רק', 'כי', 'אם', 'כאשר', 'אחרי', 'לפני', 'בתוך', 'מחוץ', 'עד',
            'כל', 'כלל', 'רוב', 'חלק', 'מעט', 'הרבה', 'יותר', 'פחות', 'כמו', 'כך', 'ככה'
        }
        
        # Pre-compile regex patterns for better performance
        if self.low_cpu_mode:
            self.word_split_pattern = re.compile(r'\S+')
            self.cleanup_pattern = re.compile(r'[^\u0590-\u05FF\u200F\u200E]')
    
    def is_hebrew_word(self, word: str) -> bool:
        """Check if word contains Hebrew characters"""
        return bool(self.hebrew_pattern.search(word))
    
    def clean_word(self, word: str) -> str:
        """Clean and normalize Hebrew word"""
        # Remove punctuation and normalize
        if self.low_cpu_mode:
            cleaned = self.cleanup_pattern.sub('', word)
        else:
            cleaned = re.sub(r'[^\u0590-\u05FF\u200F\u200E]', '', word)
        return cleaned.strip()
    
    def extract_hebrew_words(self, text: str) -> List[str]:
        """Extract Hebrew words from text"""
        if not text:
            return []
        
        # Split by whitespace and punctuation
        if self.low_cpu_mode:
            words = self.word_split_pattern.findall(text)
        else:
            words = re.findall(r'\S+', text)
        
        hebrew_words = []
        
        for word in words:
            if self.is_hebrew_word(word):
                cleaned = self.clean_word(word)
                if cleaned and len(cleaned) > 1:  # Filter out single characters
                    hebrew_words.append(cleaned)
        
        return hebrew_words
    
    def get_top_words(self, words: List[str], top_n: int = 1000, 
                     exclude_stop_words: bool = True) -> List[Tuple[str, int]]:
        """Get top N most frequent Hebrew words"""
        if exclude_stop_words:
            words = [w for w in words if w not in self.stop_words]
        
        word_counts = Counter(words)
        return word_counts.most_common(top_n)


class OpenAIWhisperProcessor:
    """Processes audio files using OpenAI Whisper API"""
    
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.supported_formats = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
        
        logger.info(f"Initialized OpenAI Whisper with model: {model}")
    
    def validate_audio_file(self, file_path: Path) -> bool:
        """Validate audio file format and size"""
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported format: {file_path.suffix}")
            return False
        
        # Check file size (OpenAI has 25MB limit)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            logger.error(f"File too large: {file_size_mb:.1f}MB (max 25MB)")
            return False
        
        logger.debug(f"File validation passed: {file_path.name} ({file_size_mb:.1f}MB)")
        return True
    
    def transcribe_audio(self, audio_path: str, language: str = "he") -> Dict:
        """Transcribe audio file using OpenAI Whisper API"""
        file_path = Path(audio_path)
        
        try:
            if not self.validate_audio_file(file_path):
                return {"text": "", "error": "File validation failed"}
            
            logger.info(f"Transcribing: {file_path.name}")
            
            # Open audio file
            with open(file_path, "rb") as audio_file:
                # Transcribe using OpenAI API
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                    response_format="verbose_json"
                )
            
            result = {
                "text": transcript.text,
                "language": transcript.language,
                "duration": transcript.duration,
                "segments": getattr(transcript, 'segments', [])
            }
            
            logger.info(f"Transcription completed: {len(transcript.text)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Failed to transcribe {file_path.name}: {e}")
            return {"text": "", "error": str(e)}


class AudioBatchProcessor:
    """Processes batches of audio files and generates word statistics"""
    
    def __init__(self, demo_files_path: str = "demo-files", api_key: str = None, 
                 low_cpu_mode: bool = False):
        self.demo_files_path = Path(demo_files_path)
        self.low_cpu_mode = low_cpu_mode
        self.word_processor = HebrewWordProcessor(low_cpu_mode=low_cpu_mode)
        self.supported_formats = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
        
        # Initialize OpenAI processor
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass --api-key")
        
        self.whisper_processor = OpenAIWhisperProcessor(api_key)
        
        # Results storage
        self.results = {}
    
    def find_audio_files(self, directory: Path) -> List[Path]:
        """Find all supported audio files in directory"""
        audio_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                audio_files.append(file_path)
        return sorted(audio_files)
    
    def process_subfolder(self, subfolder_path: Path) -> Dict:
        """Process all audio files in a subfolder"""
        logger.info(f"Processing subfolder: {subfolder_path.name}")
        
        audio_files = self.find_audio_files(subfolder_path)
        if not audio_files:
            logger.warning(f"No audio files found in {subfolder_path}")
            return {"files_processed": 0, "total_words": 0, "top_words": []}
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        all_words = []
        processed_files = 0
        failed_files = 0
        total_cost = 0.0
        
        for audio_file in tqdm(audio_files, desc=f"Processing {subfolder_path.name}"):
            try:
                # Transcribe audio
                result = self.whisper_processor.transcribe_audio(str(audio_file))
                
                if result and result.get("text") and "error" not in result:
                    # Extract Hebrew words
                    hebrew_words = self.word_processor.extract_hebrew_words(result["text"])
                    all_words.extend(hebrew_words)
                    processed_files += 1
                    
                    # Estimate cost (rough calculation)
                    duration = result.get("duration", 0)
                    cost = duration * 0.006 / 60  # $0.006 per minute
                    total_cost += cost
                    
                    logger.info(f"Processed {audio_file.name}: {len(hebrew_words)} Hebrew words, {duration:.1f}s, ~${cost:.4f}")
                else:
                    logger.warning(f"No transcription result for {audio_file.name}: {result.get('error', 'Unknown error')}")
                    failed_files += 1
                
                # Rate limiting - small delay between requests
                if self.low_cpu_mode:
                    time.sleep(1)  # 1 second delay in low CPU mode
                else:
                    time.sleep(0.5)  # 0.5 second delay in normal mode
                    
            except Exception as e:
                logger.error(f"Failed to process {audio_file.name}: {e}")
                failed_files += 1
        
        # Calculate top words
        top_words = self.word_processor.get_top_words(all_words, top_n=1000)
        
        result = {
            "subfolder": subfolder_path.name,
            "files_processed": processed_files,
            "files_failed": failed_files,
            "total_hebrew_words": len(all_words),
            "unique_hebrew_words": len(set(all_words)),
            "top_1000_words": top_words,
            "estimated_cost": round(total_cost, 4)
        }
        
        logger.info(f"Completed {subfolder_path.name}: {processed_files} files, {len(all_words)} words, ~${total_cost:.4f}")
        return result
    
    def process_all_subfolders(self) -> Dict:
        """Process all subfolders in demo-files directory"""
        if not self.demo_files_path.exists():
            raise FileNotFoundError(f"Demo files directory not found: {self.demo_files_path}")
        
        try:
            # Find all subdirectories
            subfolders = [d for d in self.demo_files_path.iterdir() if d.is_dir()]
            
            if not subfolders:
                # If no subfolders, process files directly in the directory
                logger.info("No subfolders found, processing files directly in directory")
                result = self.process_subfolder(self.demo_files_path)
                self.results[self.demo_files_path.name] = result
                return self.results
            
            logger.info(f"Found {len(subfolders)} subfolders to process")
            
            # Process each subfolder
            for subfolder in subfolders:
                try:
                    result = self.process_subfolder(subfolder)
                    self.results[subfolder.name] = result
                except Exception as e:
                    logger.error(f"Failed to process subfolder {subfolder.name}: {e}")
                    self.results[subfolder.name] = {
                        "error": str(e),
                        "files_processed": 0,
                        "total_words": 0,
                        "top_words": [],
                        "estimated_cost": 0.0
                    }
            
            return self.results
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def save_results(self, output_file: str = "hebrew_word_analysis_openai.json"):
        """Save results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def save_csv_results(self, output_file: str = "hebrew_word_analysis_openai.csv"):
        """Save results to CSV format for easy analysis"""
        try:
            rows = []
            for subfolder, data in self.results.items():
                if "error" in data:
                    continue
                    
                for rank, (word, count) in enumerate(data["top_1000_words"], 1):
                    rows.append({
                        "subfolder": subfolder,
                        "rank": rank,
                        "word": word,
                        "count": count,
                        "total_files": data["files_processed"],
                        "total_words": data["total_hebrew_words"],
                        "estimated_cost": data.get("estimated_cost", 0.0)
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"CSV results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process Hebrew audio files with OpenAI Whisper API")
    parser.add_argument("--demo-files", default="demo-files", 
                       help="Path to demo-files directory")
    parser.add_argument("--output-json", default="hebrew_word_analysis_openai.json",
                       help="Output JSON file path")
    parser.add_argument("--output-csv", default="hebrew_word_analysis_openai.csv",
                       help="Output CSV file path")
    parser.add_argument("--api-key", 
                       help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="whisper-1",
                       help="OpenAI Whisper model to use")
    parser.add_argument("--language", default="he",
                       help="Language code for transcription")
    parser.add_argument("--low-cpu", action="store_true",
                       help="Enable low CPU mode with rate limiting")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = AudioBatchProcessor(
            args.demo_files, 
            api_key=args.api_key,
            low_cpu_mode=args.low_cpu
        )
        
        # Process all subfolders
        logger.info("Starting Hebrew audio processing with OpenAI Whisper...")
        results = processor.process_all_subfolders()
        
        # Save results
        processor.save_results(args.output_json)
        processor.save_csv_results(args.output_csv)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*50)
        
        total_cost = 0.0
        for subfolder, data in results.items():
            if "error" in data:
                logger.info(f"{subfolder}: ERROR - {data['error']}")
            else:
                logger.info(f"{subfolder}:")
                logger.info(f"  Files processed: {data['files_processed']}")
                logger.info(f"  Total Hebrew words: {data['total_hebrew_words']}")
                logger.info(f"  Unique Hebrew words: {data['unique_hebrew_words']}")
                logger.info(f"  Estimated cost: ${data.get('estimated_cost', 0.0):.4f}")
                logger.info(f"  Top word: {data['top_1000_words'][0] if data['top_1000_words'] else 'N/A'}")
                total_cost += data.get('estimated_cost', 0.0)
        
        logger.info(f"\nTotal estimated cost: ${total_cost:.4f}")
        logger.info(f"\nResults saved to:")
        logger.info(f"  JSON: {args.output_json}")
        logger.info(f"  CSV: {args.output_csv}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
