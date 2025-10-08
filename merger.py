#!/usr/bin/env python3
"""
Script to merge and analyze data from results subfolders and create Excel files with translations.

This script processes CSV files from results-100, results-1000, and results-5000 directories,
combines data from radio, tech-meetings, and tv categories, and creates Excel files with
translations for specified languages.

Features:
- Combines word frequency data from multiple categories (radio, tech-meetings, tv)
- Aggregates counts and calculates statistics (avg_rank, min_rank, max_rank)
- Supports translation columns for English and Russian
- Prioritizes existing translations from CSV files over built-in dictionary
- Optional OpenAI integration for missing translations with caching
- Creates Excel files with auto-adjusted column widths
- Handles empty translation columns gracefully

Usage Examples:
    python merger.py --additional_langs=eng,rus    # Both English and Russian translations
    python merger.py --additional_langs=eng        # Only English translations
    python merger.py --additional_langs=rus        # Only Russian translations
    python merger.py --additional_langs=eng,rus --use_openai  # Use OpenAI for missing translations
    python merger.py --additional_langs=eng --use_openai --cache_save_interval=10  # Save cache every 10 translations
    python merger.py --additional_langs=eng --separate  # Create separate files for each category
    python merger.py                               # No additional translations

Output Files:
    - results-100.xlsx    (Top 100 combined words)
    - results-1000.xlsx   (Top 1000 combined words)
    - results-5000.xlsx   (Top 5000 combined words)
    
    With --separate flag, also creates:
    - results-100_radio.xlsx, results-100_tech-meetings.xlsx, results-100_tv.xlsx
    - results-1000_radio.xlsx, results-1000_tech-meetings.xlsx, results-1000_tv.xlsx
    - results-5000_radio.xlsx, results-5000_tech-meetings.xlsx, results-5000_tv.xlsx

Excel Columns:
    - id: Sequential ID
    - rank: Rank based on total count
    - word: Hebrew word
    - count: Total count across all categories
    - categories: Categories where word appears (comma-separated)
    - avg_rank: Average rank across categories
    - min_rank: Minimum rank across categories
    - max_rank: Maximum rank across categories
    - eng: English translation (if --additional_langs=eng)
    - rus: Russian translation (if --additional_langs=rus)
"""

import os
import csv
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, Counter
import logging
import json
import time
import signal
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataMerger:
    """
    Class to handle merging and processing of word analysis data from multiple sources.
    """
    
    def __init__(self, additional_langs: List[str] = None, use_openai: bool = False, cache_save_interval: int = 20, separate: bool = False):
        """
        Initialize the DataMerger.
        
        Args:
            additional_langs: List of additional languages for translation (e.g., ['eng', 'rus'])
            use_openai: Whether to use OpenAI for missing translations
            cache_save_interval: Save cache every N new translations (default: 20)
            separate: Whether to create separate Excel files for each category
        """
        self.additional_langs = additional_langs or []
        self.use_openai = use_openai
        self.cache_save_interval = cache_save_interval
        self.separate = separate
        self.results_base_dir = 'results'
        self.categories = ['radio', 'tech-meetings', 'tv']
        self.result_types = ['results-100', 'results-1000', 'results-5000']
        
        # Initialize OpenAI client if needed
        self.openai_client = None
        if self.use_openai:
            self._init_openai_client()
        
        # Translation cache to avoid duplicate API calls
        self.translation_cache = {}
        self.cache_file = 'translation_cache.json'
        self.new_translations_count = 0
        self._load_translation_cache()
        
        # Set up signal handler to save cache on interruption
        if self.use_openai:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Extended translation dictionaries for common Hebrew words
        self.translations = {
            'eng': {
                # Basic words
                'לא': 'no', 'לי': 'to me', 'עוד': 'more', 'יש': 'there is', 'מה': 'what',
                'לך': 'to you', 'שלי': 'mine', 'חיי': 'my life', 'באמת': 'really',
                'אז': 'so', 'כן': 'yes', 'אוקיי': 'okay', 'אני': 'I', 'אתה': 'you',
                'הוא': 'he', 'היא': 'she', 'אנחנו': 'we', 'אתם': 'you (plural)',
                'הם': 'they', 'זה': 'this', 'זאת': 'this (fem)', 'אלה': 'these',
                'כל': 'all', 'אין': 'there is not', 'רק': 'only', 'גם': 'also',
                'עכשיו': 'now', 'אחרי': 'after', 'לפני': 'before', 'בין': 'between',
                'על': 'on', 'תחת': 'under', 'ליד': 'near', 'בתוך': 'inside',
                'מחוץ': 'outside', 'מעל': 'above', 'מתחת': 'below', 'אצל': 'at',
                'עם': 'with', 'בלי': 'without', 'בשביל': 'for', 'בגלל': 'because',
                'אבל': 'but', 'או': 'or', 'כי': 'because', 'אם': 'if', 'כאשר': 'when',
                
                # Additional common words
                'הזה': 'this', 'פה': 'here', 'טוב': 'good', 'מאוד': 'very', 'רוצה': 'want',
                'שם': 'there', 'לנו': 'to us', 'כבר': 'already', 'שזה': 'that this',
                'צריך': 'need', 'יכול': 'can', 'הולך': 'going', 'בא': 'coming',
                'עושה': 'doing', 'אומר': 'saying', 'יודע': 'knowing', 'חושב': 'thinking',
                'רואה': 'seeing', 'שומע': 'hearing', 'אוהב': 'loving', 'חי': 'living',
                'עובד': 'working', 'לומד': 'studying', 'מדבר': 'speaking', 'כותב': 'writing',
                'קורא': 'reading', 'שואל': 'asking', 'עונה': 'answering', 'מבין': 'understanding',
                'זוכר': 'remembering', 'שוכח': 'forgetting', 'מתחיל': 'starting', 'מסיים': 'finishing',
                'פותח': 'opening', 'סוגר': 'closing', 'נותן': 'giving', 'לוקח': 'taking',
                'קונה': 'buying', 'מוכר': 'selling', 'שולח': 'sending', 'מקבל': 'receiving',
                'בונה': 'building', 'הורס': 'destroying', 'יוצר': 'creating', 'משנה': 'changing',
                'מתקן': 'fixing', 'שובר': 'breaking', 'מתחבר': 'connecting', 'מתנתק': 'disconnecting',
                'מתחיל': 'beginning', 'מסיים': 'ending', 'ממשיך': 'continuing', 'עוצר': 'stopping',
                'מתקדם': 'advancing', 'נסוג': 'retreating', 'עולה': 'going up', 'יורד': 'going down',
                'נכנס': 'entering', 'יוצא': 'exiting', 'מגיע': 'arriving', 'עוזב': 'leaving',
                'נשאר': 'staying', 'הולך': 'walking', 'רץ': 'running', 'עף': 'flying',
                'שח': 'swimming', 'קופץ': 'jumping', 'נופל': 'falling', 'עומד': 'standing',
                'יושב': 'sitting', 'שוכב': 'lying', 'ישן': 'sleeping', 'מתעורר': 'waking up',
                'אוכל': 'eating', 'שותה': 'drinking', 'רואה': 'seeing', 'שומע': 'hearing',
                'מריח': 'smelling', 'טועם': 'tasting', 'נוגע': 'touching', 'מרגיש': 'feeling'
            },
            'rus': {
                # Basic words
                'לא': 'нет', 'לי': 'мне', 'עוד': 'еще', 'יש': 'есть', 'מה': 'что',
                'לך': 'тебе', 'שלי': 'мой', 'חיי': 'моя жизнь', 'באמת': 'действительно',
                'אז': 'так', 'כן': 'да', 'אוקיי': 'окей', 'אני': 'я', 'אתה': 'ты',
                'הוא': 'он', 'היא': 'она', 'אנחנו': 'мы', 'אתם': 'вы',
                'הם': 'они', 'זה': 'это', 'זאת': 'это (ж.р.)', 'אלה': 'эти',
                'כל': 'все', 'אין': 'нет', 'רק': 'только', 'גם': 'тоже',
                'עכשיו': 'сейчас', 'אחרי': 'после', 'לפני': 'перед', 'בין': 'между',
                'על': 'на', 'תחת': 'под', 'ליד': 'рядом', 'בתוך': 'внутри',
                'מחוץ': 'снаружи', 'מעל': 'над', 'מתחת': 'под', 'אצל': 'у',
                'עם': 'с', 'בלי': 'без', 'בשביל': 'для', 'בגלל': 'из-за',
                'אבל': 'но', 'או': 'или', 'כי': 'потому что', 'אם': 'если', 'כאשר': 'когда',
                
                # Additional common words
                'הזה': 'этот', 'פה': 'здесь', 'טוב': 'хорошо', 'מאוד': 'очень', 'רוצה': 'хочу',
                'שם': 'там', 'לנו': 'нам', 'כבר': 'уже', 'שזה': 'что это',
                'צריך': 'нужно', 'יכול': 'могу', 'הולך': 'иду', 'בא': 'прихожу',
                'עושה': 'делаю', 'אומר': 'говорю', 'יודע': 'знаю', 'חושב': 'думаю',
                'רואה': 'вижу', 'שומע': 'слышу', 'אוהב': 'люблю', 'חי': 'живу',
                'עובד': 'работаю', 'לומד': 'учусь', 'מדבר': 'говорю', 'כותב': 'пишу',
                'קורא': 'читаю', 'שואל': 'спрашиваю', 'עונה': 'отвечаю', 'מבין': 'понимаю',
                'זוכר': 'помню', 'שוכח': 'забываю', 'מתחיל': 'начинаю', 'מסיים': 'заканчиваю',
                'פותח': 'открываю', 'סוגר': 'закрываю', 'נותן': 'даю', 'לוקח': 'беру',
                'קונה': 'покупаю', 'מוכר': 'продаю', 'שולח': 'отправляю', 'מקבל': 'получаю',
                'בונה': 'строю', 'הורס': 'разрушаю', 'יוצר': 'создаю', 'משנה': 'меняю',
                'מתקן': 'чиню', 'שובר': 'ломаю', 'מתחבר': 'подключаюсь', 'מתנתק': 'отключаюсь',
                'מתחיל': 'начинаю', 'מסיים': 'заканчиваю', 'ממשיך': 'продолжаю', 'עוצר': 'останавливаюсь',
                'מתקדם': 'продвигаюсь', 'נסוג': 'отступаю', 'עולה': 'поднимаюсь', 'יורד': 'спускаюсь',
                'נכנס': 'вхожу', 'יוצא': 'выхожу', 'מגיע': 'прибываю', 'עוזב': 'ухожу',
                'נשאר': 'остаюсь', 'הולך': 'иду', 'רץ': 'бегу', 'עף': 'лечу',
                'שח': 'плыву', 'קופץ': 'прыгаю', 'נופל': 'падаю', 'עומד': 'стою',
                'יושב': 'сижу', 'שוכב': 'лежу', 'ישן': 'сплю', 'מתעורר': 'просыпаюсь',
                'אוכל': 'ем', 'שותה': 'пью', 'רואה': 'вижу', 'שומע': 'слышу',
                'מריח': 'нюхаю', 'טועם': 'пробую', 'נוגע': 'трогаю', 'מרגיש': 'чувствую'
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals to save cache before exiting."""
        logger.info(f"Received signal {signum}, saving translation cache...")
        if self.translation_cache:
            self._save_translation_cache()
        logger.info("Cache saved. Exiting...")
        sys.exit(0)
    
    def _init_openai_client(self) -> None:
        """Initialize OpenAI client with API key."""
        try:
            # Load environment variables
            load_dotenv()
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable or use setup_openai.py")
                self.use_openai = False
                return
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.use_openai = False
    
    def _load_translation_cache(self) -> None:
        """Load translation cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.translation_cache = json.load(f)
                logger.info(f"Loaded {len(self.translation_cache)} cached translations")
        except Exception as e:
            logger.warning(f"Could not load translation cache: {e}")
            self.translation_cache = {}
    
    def _save_translation_cache(self) -> None:
        """Save translation cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.translation_cache)} total translations to cache ({self.new_translations_count} new this session)")
        except Exception as e:
            logger.warning(f"Could not save translation cache: {e}")
    
    def _get_openai_translation(self, word: str, target_lang: str) -> str:
        """
        Get translation from OpenAI API.
        
        Args:
            word: Hebrew word to translate
            target_lang: Target language code ('eng' or 'rus')
            
        Returns:
            Translation or empty string if failed
        """
        if not self.openai_client:
            return ""
        
        # Check cache first
        cache_key = f"{word}_{target_lang}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Map language codes to full names
            lang_map = {'eng': 'English', 'rus': 'Russian'}
            target_language = lang_map.get(target_lang, target_lang)
            
            # Create prompt for translation
            prompt = f"Translate the Hebrew word '{word}' to {target_language}. Return only the translation, no explanations or additional text."
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a Hebrew translator. Provide accurate, concise translations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            translation = response.choices[0].message.content.strip()
            
            # Cache the result
            self.translation_cache[cache_key] = translation
            self.new_translations_count += 1
            
            # Save cache every 20 new translations
            if self.new_translations_count % self.cache_save_interval == 0:
                self._save_translation_cache()
                logger.info(f"Saved cache after {self.new_translations_count} new translations")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
            
            return translation
            
        except Exception as e:
            logger.warning(f"OpenAI translation failed for '{word}' to {target_lang}: {e}")
            return ""
    
    def load_csv_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of dictionaries containing the CSV data
        """
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Clean up empty values in translation columns
                    if 'russian_translation' in row and not row['russian_translation'].strip():
                        row['russian_translation'] = ''
                    data.append(row)
            logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return []
    
    def get_translation(self, word: str, lang: str, existing_translations: Dict[str, str] = None) -> str:
        """
        Get translation for a word in the specified language.
        Prioritizes existing translations from CSV files, then built-in dictionary, then OpenAI.
        
        Args:
            word: Hebrew word to translate
            lang: Target language code ('eng' or 'rus')
            existing_translations: Dictionary of existing translations from CSV files
            
        Returns:
            Translation of the word, or empty string if not found
        """
        # First check existing translations from CSV files
        if existing_translations and lang in existing_translations:
            return existing_translations[lang]
        
        # Fall back to built-in dictionary
        if lang in self.translations and word in self.translations[lang]:
            return self.translations[lang][word]
        
        # Use OpenAI for missing translations if enabled
        if self.use_openai:
            openai_translation = self._get_openai_translation(word, lang)
            if openai_translation:
                return openai_translation
        
        return ""
    
    def combine_data_from_categories(self, result_type: str) -> List[Dict[str, Any]]:
        """
        Combine data from all categories (radio, tech-meetings, tv) for a specific result type.
        
        Args:
            result_type: Type of results ('results-100', 'results-1000', 'results-5000')
            
        Returns:
            Combined list of word data with aggregated counts
        """
        combined_data = defaultdict(lambda: {
            'word': '',
            'total_count': 0,
            'categories': set(),
            'ranks': [],
            'existing_translations': {}
        })
        
        for category in self.categories:
            # Determine the correct filename based on result type and category
            if result_type == 'results-5000':
                if category == 'radio':
                    filename = 'radio-5000.csv'
                elif category == 'tech-meetings':
                    filename = 'tech-meetings-5000.csv'
                else:  # tv
                    filename = 'hebrew_word_analysis_openai.csv'
            else:
                filename = 'hebrew_word_analysis_openai.csv'
            
            file_path = os.path.join(self.results_base_dir, result_type, category, filename)
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            data = self.load_csv_data(file_path)
            
            for row in data:
                word = row['word']
                count = int(row['count'])
                rank = int(row['rank'])
                
                combined_data[word]['word'] = word
                combined_data[word]['total_count'] += count
                combined_data[word]['categories'].add(category)
                combined_data[word]['ranks'].append(rank)
                
                # Store existing translations from CSV files
                if 'russian_translation' in row and row['russian_translation'].strip():
                    combined_data[word]['existing_translations']['rus'] = row['russian_translation'].strip()
        
        # Convert to list and sort by total count
        result = []
        for word, data in combined_data.items():
            result.append({
                'word': word,
                'total_count': data['total_count'],
                'categories': ', '.join(sorted(data['categories'])),
                'avg_rank': sum(data['ranks']) / len(data['ranks']) if data['ranks'] else 0,
                'min_rank': min(data['ranks']) if data['ranks'] else 0,
                'max_rank': max(data['ranks']) if data['ranks'] else 0,
                'existing_translations': data['existing_translations']
            })
        
        # Sort by total count (descending)
        result.sort(key=lambda x: x['total_count'], reverse=True)
        
        logger.info(f"Combined {len(result)} unique words from {result_type}")
        return result
    
    def combine_data_from_single_category(self, result_type: str, category: str) -> List[Dict[str, Any]]:
        """
        Combine data from a single category for a specific result type.
        
        Args:
            result_type: Type of results ('results-100', 'results-1000', 'results-5000')
            category: Category name ('radio', 'tech-meetings', 'tv')
            
        Returns:
            List of word data for the single category
        """
        # Determine the correct filename based on result type and category
        if result_type == 'results-5000':
            if category == 'radio':
                filename = 'radio-5000.csv'
            elif category == 'tech-meetings':
                filename = 'tech-meetings-5000.csv'
            else:  # tv
                filename = 'hebrew_word_analysis_openai.csv'
        else:
            filename = 'hebrew_word_analysis_openai.csv'
        
        file_path = os.path.join(self.results_base_dir, result_type, category, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return []
        
        data = self.load_csv_data(file_path)
        
        # Process data for single category
        result = []
        for idx, row in enumerate(data, 1):
            word = row['word']
            count = int(row['count'])
            rank = int(row['rank'])
            
            # Get existing translations
            existing_translations = {}
            if 'russian_translation' in row and row['russian_translation'].strip():
                existing_translations['rus'] = row['russian_translation'].strip()
            
            result.append({
                'word': word,
                'total_count': count,
                'categories': category,
                'avg_rank': rank,
                'min_rank': rank,
                'max_rank': rank,
                'existing_translations': existing_translations
            })
        
        # Sort by count (descending)
        result.sort(key=lambda x: x['total_count'], reverse=True)
        
        logger.info(f"Processed {len(result)} words from {result_type}/{category}")
        return result
    
    def calculate_translation_coverage(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """
        Calculate translation coverage statistics.
        
        Args:
            data: Combined word data
            
        Returns:
            Dictionary with coverage statistics for each language
        """
        coverage = {}
        total_words = len(data)
        
        for lang in self.additional_langs:
            translated_count = 0
            for item in data:
                translation = self.get_translation(item['word'], lang, item.get('existing_translations', {}))
                if translation and translation.strip():
                    translated_count += 1
            
            coverage[lang] = {
                'translated': translated_count,
                'total': total_words,
                'percentage': round((translated_count / total_words) * 100, 1) if total_words > 0 else 0
            }
        
        return coverage
    
    def create_excel_output(self, data: List[Dict[str, Any]], result_type: str) -> None:
        """
        Create Excel file with the combined data and translations.
        
        Args:
            data: Combined word data
            result_type: Type of results for filename
        """
        # Calculate translation coverage
        coverage = self.calculate_translation_coverage(data)
        
        # Log translation coverage statistics
        for lang in self.additional_langs:
            if lang in coverage:
                stats = coverage[lang]
                logger.info(f"Translation coverage for {lang}: {stats['translated']}/{stats['total']} ({stats['percentage']}%)")
        
        # Log some untranslated words for reference
        untranslated_words = []
        for item in data[:20]:  # Check first 20 words
            has_translation = False
            for lang in self.additional_langs:
                translation = self.get_translation(item['word'], lang, item.get('existing_translations', {}))
                if translation and translation.strip():
                    has_translation = True
                    break
            if not has_translation:
                untranslated_words.append(item['word'])
        
        if untranslated_words:
            logger.info(f"Sample untranslated words: {', '.join(untranslated_words[:10])}")
        
        # Prepare data for Excel
        excel_data = []
        
        for idx, item in enumerate(data, 1):
            row = {
                'id': idx,
                'rank': idx,
                'word': item['word'],
                'count': item['total_count'],
                'categories': item['categories'],
                'avg_rank': round(item['avg_rank'], 2),
                'min_rank': item['min_rank'],
                'max_rank': item['max_rank']
            }
            
            # Add translations for requested languages
            for lang in self.additional_langs:
                translation = self.get_translation(item['word'], lang, item.get('existing_translations', {}))
                # Convert NaN to empty string for better Excel display
                if pd.isna(translation) or translation is None:
                    translation = ''
                row[lang] = translation
            
            excel_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(excel_data)
        
        # Create output filename
        if result_type.endswith('.xlsx'):
            output_filename = result_type
        else:
            output_filename = f"{result_type}.xlsx"
        
        # Write to Excel
        try:
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Word Analysis', index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets['Word Analysis']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            logger.info(f"Created Excel file: {output_filename} with {len(excel_data)} records")
            
        except Exception as e:
            logger.error(f"Error creating Excel file {output_filename}: {e}")
    
    def process_all_result_types(self) -> None:
        """
        Process all result types and create corresponding Excel files.
        """
        logger.info("Starting data merger process...")
        logger.info(f"Additional languages: {', '.join(self.additional_langs) if self.additional_langs else 'None'}")
        
        for result_type in self.result_types:
            logger.info(f"Processing {result_type}...")
            
            # Check if the result type directory exists
            result_dir = os.path.join(self.results_base_dir, result_type)
            if not os.path.exists(result_dir):
                logger.warning(f"Directory {result_dir} does not exist, skipping...")
                continue
            
            # Combine data from all categories
            combined_data = self.combine_data_from_categories(result_type)
            
            if not combined_data:
                logger.warning(f"No data found for {result_type}, skipping...")
                continue
            
            # Create Excel output for combined data
            self.create_excel_output(combined_data, result_type)
            
            # Create separate Excel files for each category if requested
            if self.separate:
                for category in self.categories:
                    category_data = self.combine_data_from_single_category(result_type, category)
                    if category_data:
                        category_filename = f"{result_type}_{category}"
                        self.create_excel_output(category_data, category_filename)
            
            # Save translation cache after each result type to avoid losing progress
            if self.use_openai and self.translation_cache:
                self._save_translation_cache()
        
        # Final save of translation cache
        if self.use_openai and self.translation_cache:
            self._save_translation_cache()
        
        logger.info("Data merger process completed!")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Merge word analysis data from results subfolders and create Excel files with translations'
    )
    
    parser.add_argument(
        '--additional_langs',
        type=str,
        default='',
        help='Comma-separated list of additional languages for translation (e.g., eng,rus)'
    )
    
    parser.add_argument(
        '--use_openai',
        action='store_true',
        help='Use OpenAI API for missing translations (requires OPENAI_API_KEY)'
    )
    
    parser.add_argument(
        '--cache_save_interval',
        type=int,
        default=20,
        help='Save translation cache every N new translations (default: 20)'
    )
    
    parser.add_argument(
        '--separate',
        action='store_true',
        help='Create separate Excel files for each category (radio, tech-meetings, tv)'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the data merger.
    """
    args = parse_arguments()
    
    # Parse additional languages
    additional_langs = []
    if args.additional_langs:
        additional_langs = [lang.strip().lower() for lang in args.additional_langs.split(',')]
        # Validate language codes
        valid_langs = ['eng', 'rus']
        invalid_langs = [lang for lang in additional_langs if lang not in valid_langs]
        if invalid_langs:
            logger.error(f"Invalid language codes: {', '.join(invalid_langs)}. Valid codes: {', '.join(valid_langs)}")
            return
    
    # Create and run the merger
    merger = DataMerger(
        additional_langs=additional_langs, 
        use_openai=args.use_openai,
        cache_save_interval=args.cache_save_interval,
        separate=args.separate
    )
    merger.process_all_result_types()


if __name__ == "__main__":
    main()
