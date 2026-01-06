#!/usr/bin/env python3
"""
JSON to TTS Text Utility
========================
Converts the structured JSON output from Dolphin into a clean, 
continuous text file optimized for Text-to-Speech (TTS).

Features:
- Filters out headers and footers (page numbers, book titles)
- Keeps chapter headings and story text
- Cleans up OCR artifacts (hyphenation, extra spaces)
- Joins text into a flowing narrative
"""

import json
import re
import argparse
import sys
from pathlib import Path

def clean_element_text(text):
    """Clean individual element text."""
    if not text:
        return ""
    
    # Remove LaTeX markers if any left
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    
    # Handle hyphenation at line breaks within an element
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # Replace single newlines with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_storybook_json(json_data):
    """Extract and clean story text from JSON."""
    pages = json_data.get("pages", [])
    story_blocks = []
    
    # Labels we want to EXCLUDE from the audio version
    EXCLUDE_LABELS = {'header', 'foot', 'page_num', 'distorted_page'}
    
    for page in pages:
        page_elements = page.get("elements", [])
        # Sort by reading order just in case
        page_elements.sort(key=lambda x: x.get("reading_order", 0))
        
        for elem in page_elements:
            label = elem.get("label", "").lower()
            content = elem.get("text", "")
            
            if label not in EXCLUDE_LABELS and content:
                cleaned = clean_element_text(content)
                if cleaned:
                    story_blocks.append(cleaned)
    
    # Join everything with double newlines for paragraph separation
    return "\n\n".join(story_blocks)

def main():
    parser = argparse.ArgumentParser(description="Convert Dolphin JSON to clean TTS-ready text")
    parser.add_argument("input", help="Input JSON file (e.g., wizard_storybook.json)")
    parser.add_argument("-o", "--output", help="Output text file (default: input_name.txt)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} not found.")
        sys.exit(1)
        
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        sys.exit(1)
        
    print(f"📖 Processing {input_path.name}...")
    
    story_text = process_storybook_json(data)
    
    output_path = args.output or input_path.with_suffix('.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(story_text)
        
    print(f"✅ Success! TTS-ready text saved to {output_path}")
    print(f"📊 Total characters: {len(story_text)}")

if __name__ == "__main__":
    main()
