#!/usr/bin/env python3
"""
Hebrew Audio Processing Script with WhisperX
Processes audio files in demo-files subdirectories and calculates top 1000 Hebrew words
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
import gc
import torch

# WhisperX imports
import whisperx
import torchaudio

# Data processing
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_processing.log'),
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
        # Remove diacritics (optional - can be enabled for more aggressive normalization)
        # cleaned = re.sub(r'[\u0591-\u05C7]', '', cleaned)
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


class AudioProcessor:
    """Processes audio files using WhisperX"""
    
    def __init__(self, device: str = "cuda", compute_type: str = "float16", 
                 low_cpu_mode: bool = False):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.low_cpu_mode = low_cpu_mode
        self.model = None
        self.align_model = None
        self.metadata = None
        
        # CPU optimization settings
        if self.low_cpu_mode:
            self.compute_type = "int8"  # Use int8 for lower CPU usage
            logger.info("Low CPU mode enabled - using int8 compute type")
        
        logger.info(f"Using device: {self.device}, compute_type: {self.compute_type}")
    
    def load_models(self, model_size: str = "large-v2"):
        """Load WhisperX models"""
        try:
            logger.info(f"Loading WhisperX model: {model_size}")
            self.model = whisperx.load_model(
                model_size, 
                self.device, 
                compute_type=self.compute_type
            )
            logger.info("WhisperX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            raise
    
    def load_alignment_model(self, language_code: str = "he"):
        """Load alignment model for Hebrew"""
        try:
            logger.info(f"Loading alignment model for language: {language_code}")
            self.align_model, self.metadata = whisperx.load_align_model(
                language_code=language_code, 
                device=self.device
            )
            logger.info("Alignment model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load alignment model: {e}")
            logger.info("Continuing without alignment (timestamps may be less accurate)")
            self.align_model = None
            self.metadata = None
    
    def transcribe_audio(self, audio_path: str, batch_size: int = 16) -> Dict:
        """Transcribe audio file using WhisperX"""
        try:
            logger.info(f"Transcribing: {audio_path}")
            
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Adjust batch size for low CPU mode
            if self.low_cpu_mode:
                batch_size = min(batch_size, 8)  # Reduce batch size for lower CPU usage
                logger.debug(f"Low CPU mode: using batch_size={batch_size}")
            
            # Transcribe
            result = self.model.transcribe(audio, batch_size=batch_size)
            
            # Align if model is available (skip in low CPU mode for speed)
            if self.align_model and self.metadata and not self.low_cpu_mode:
                logger.info("Performing word-level alignment")
                result = whisperx.align(
                    result["segments"], 
                    self.align_model, 
                    self.metadata, 
                    audio, 
                    self.device, 
                    return_char_alignments=False
                )
            elif self.low_cpu_mode:
                logger.debug("Skipping alignment in low CPU mode")
            
            # Clear audio from memory immediately
            del audio
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_path}: {e}")
            return {"segments": [], "text": ""}
    
    def cleanup_models(self):
        """Clean up GPU memory"""
        if self.model:
            del self.model
        if self.align_model:
            del self.align_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AudioBatchProcessor:
    """Processes batches of audio files and generates word statistics"""
    
    def __init__(self, demo_files_path: str = "demo-files", low_cpu_mode: bool = False):
        self.demo_files_path = Path(demo_files_path)
        self.low_cpu_mode = low_cpu_mode
        self.audio_processor = AudioProcessor(low_cpu_mode=low_cpu_mode)
        self.word_processor = HebrewWordProcessor(low_cpu_mode=low_cpu_mode)
        self.supported_formats = {'.mp3', '.ogg', '.m4a', '.wav', '.flac'}
        
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
        
        # Process files with memory optimization
        for audio_file in tqdm(audio_files, desc=f"Processing {subfolder_path.name}"):
            try:
                # Transcribe audio
                result = self.audio_processor.transcribe_audio(str(audio_file))
                
                if result and result.get("text"):
                    # Extract Hebrew words
                    hebrew_words = self.word_processor.extract_hebrew_words(result["text"])
                    all_words.extend(hebrew_words)
                    processed_files += 1
                    
                    logger.info(f"Processed {audio_file.name}: {len(hebrew_words)} Hebrew words")
                else:
                    logger.warning(f"No transcription result for {audio_file.name}")
                    failed_files += 1
                
                # Memory cleanup after each file in low CPU mode
                if self.low_cpu_mode:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
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
            "top_1000_words": top_words
        }
        
        logger.info(f"Completed {subfolder_path.name}: {processed_files} files, {len(all_words)} words")
        return result
    
    def process_all_subfolders(self) -> Dict:
        """Process all subfolders in demo-files directory"""
        if not self.demo_files_path.exists():
            raise FileNotFoundError(f"Demo files directory not found: {self.demo_files_path}")
        
        # Load models once
        self.audio_processor.load_models()
        
        # Skip alignment model in low CPU mode for better performance
        if not self.low_cpu_mode:
            self.audio_processor.load_alignment_model("he")  # Hebrew
        else:
            logger.info("Skipping alignment model loading in low CPU mode")
        
        try:
            # Find all subdirectories
            subfolders = [d for d in self.demo_files_path.iterdir() if d.is_dir()]
            
            if not subfolders:
                logger.warning("No subfolders found in demo-files directory")
                return {}
            
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
                        "top_words": []
                    }
            
            return self.results
            
        finally:
            # Cleanup models
            self.audio_processor.cleanup_models()
    
    def save_results(self, output_file: str = "hebrew_word_analysis.json"):
        """Save results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def save_csv_results(self, output_file: str = "hebrew_word_analysis.csv"):
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
                        "total_words": data["total_hebrew_words"]
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"CSV results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process Hebrew audio files with WhisperX")
    parser.add_argument("--demo-files", default="demo-files", 
                       help="Path to demo-files directory")
    parser.add_argument("--output-json", default="hebrew_word_analysis.json",
                       help="Output JSON file path")
    parser.add_argument("--output-csv", default="hebrew_word_analysis.csv",
                       help="Output CSV file path")
    parser.add_argument("--model-size", default="large-v2",
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="WhisperX model size")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--device", default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for processing")
    parser.add_argument("--low-cpu", action="store_true",
                       help="Enable low CPU mode for reduced resource usage")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor with low CPU mode if requested
        processor = AudioBatchProcessor(args.demo_files, low_cpu_mode=args.low_cpu)
        
        # Process all subfolders
        logger.info("Starting Hebrew audio processing...")
        results = processor.process_all_subfolders()
        
        # Save results
        processor.save_results(args.output_json)
        processor.save_csv_results(args.output_csv)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*50)
        
        for subfolder, data in results.items():
            if "error" in data:
                logger.info(f"{subfolder}: ERROR - {data['error']}")
            else:
                logger.info(f"{subfolder}:")
                logger.info(f"  Files processed: {data['files_processed']}")
                logger.info(f"  Total Hebrew words: {data['total_hebrew_words']}")
                logger.info(f"  Unique Hebrew words: {data['unique_hebrew_words']}")
                logger.info(f"  Top word: {data['top_1000_words'][0] if data['top_1000_words'] else 'N/A'}")
        
        logger.info(f"\nResults saved to:")
        logger.info(f"  JSON: {args.output_json}")
        logger.info(f"  CSV: {args.output_csv}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
