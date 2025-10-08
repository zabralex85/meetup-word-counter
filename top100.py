#!/usr/bin/env python3
"""
Script to extract top 100 results from results-1000/ and create corresponding files in results-100/

This script processes CSV and JSON files from the results-1000/ directory structure
and creates new files with only the top 100 entries in the results-100/ directory.
"""

import os
import json
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple


def create_directory_structure(base_path: str) -> None:
    """
    Create the results-100 directory structure with subdirectories.
    
    Args:
        base_path: Base path for the results-100 directory
    """
    subdirs = ['radio', 'tech-meetings', 'tv']
    
    # Create main directory
    os.makedirs(base_path, exist_ok=True)
    
    # Create subdirectories
    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        print(f"Created directory: {subdir_path}")


def process_csv_file(input_path: str, output_path: str, top_n: int = 100) -> None:
    """
    Process CSV file to extract top N entries.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        top_n: Number of top entries to extract (default: 100)
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Read header
            
            # Read all data rows
            rows = list(reader)
            
            # Take only top N rows (excluding header)
            top_rows = rows[:top_n]
            
        # Write to output file
        with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Write header
            writer.writerows(top_rows)  # Write top N rows
            
        print(f"Processed CSV: {input_path} -> {output_path} ({len(top_rows)} entries)")
        
    except Exception as e:
        print(f"Error processing CSV file {input_path}: {e}")


def process_json_file(input_path: str, output_path: str, top_n: int = 100) -> None:
    """
    Process JSON file to extract top N entries.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        top_n: Number of top entries to extract (default: 100)
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # Process each subfolder in the JSON data
        for subfolder_key, subfolder_data in data.items():
            if 'top_1000_words' in subfolder_data:
                # Take only top N words
                subfolder_data['top_1000_words'] = subfolder_data['top_1000_words'][:top_n]
                # Update the key name to reflect the actual number
                subfolder_data['top_100_words'] = subfolder_data.pop('top_1000_words')
        
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)
            
        print(f"Processed JSON: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing JSON file {input_path}: {e}")


def process_category(input_dir: str, output_dir: str, category: str, top_n: int = 100) -> None:
    """
    Process all files for a specific category.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        category: Category name (radio, tech-meetings, tv)
        top_n: Number of top entries to extract
    """
    category_input_path = os.path.join(input_dir, category)
    category_output_path = os.path.join(output_dir, category)
    
    if not os.path.exists(category_input_path):
        print(f"Warning: Input directory {category_input_path} does not exist")
        return
    
    # Process CSV file
    csv_input = os.path.join(category_input_path, 'hebrew_word_analysis_openai.csv')
    csv_output = os.path.join(category_output_path, 'hebrew_word_analysis_openai.csv')
    
    if os.path.exists(csv_input):
        process_csv_file(csv_input, csv_output, top_n)
    else:
        print(f"Warning: CSV file {csv_input} does not exist")
    
    # Process JSON file
    json_input = os.path.join(category_input_path, 'hebrew_word_analysis_openai.json')
    json_output = os.path.join(category_output_path, 'hebrew_word_analysis_openai.json')
    
    if os.path.exists(json_input):
        process_json_file(json_input, json_output, top_n)
    else:
        print(f"Warning: JSON file {json_input} does not exist")


def main():
    """
    Main function to process all categories and create top 100 results.
    """
    # Define paths
    input_base_dir = 'results/results-1000'
    output_base_dir = 'results/results-100'
    categories = ['radio', 'tech-meetings', 'tv']
    top_n = 100
    
    print(f"Starting top {top_n} extraction from {input_base_dir} to {output_base_dir}")
    print("=" * 60)
    
    # Check if input directory exists
    if not os.path.exists(input_base_dir):
        print(f"Error: Input directory {input_base_dir} does not exist")
        return
    
    # Create output directory structure
    create_directory_structure(output_base_dir)
    print()
    
    # Process each category
    for category in categories:
        print(f"Processing category: {category}")
        print("-" * 40)
        process_category(input_base_dir, output_base_dir, category, top_n)
        print()
    
    print("=" * 60)
    print(f"Top {top_n} extraction completed successfully!")
    print(f"Results saved to: {output_base_dir}")


if __name__ == "__main__":
    main()
