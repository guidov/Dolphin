#!/usr/bin/env python3
"""
Dolphin PDF Parser - CLI Tool
==============================
Parse PDFs using the Dolphin model on Modal and output structured JSON for LLMs.

Usage:
    python dolphin_cli.py book.pdf                      # Output JSON to stdout
    python dolphin_cli.py book.pdf -o book.json         # Save as JSON
    python dolphin_cli.py book.pdf -o book.md --format markdown  # Save as markdown
    python dolphin_cli.py book.pdf --text-only          # Output plain text only

Examples:
    # Process a book and save structured JSON
    python dolphin_cli.py "Charlotte's Web.pdf" -o charlottes_web.json
    
    # Get just the text for piping to another tool
    python dolphin_cli.py document.pdf --text-only | wc -w
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Parse PDFs using Dolphin AI on Modal - outputs LLM-friendly JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("input", help="Input PDF or image file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--format", 
        choices=["json", "markdown", "text"], 
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Output only the extracted text (no metadata)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output (no indentation)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Determine file type
    ext = input_path.suffix.lower()
    if ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
        print(f"Error: Unsupported file type '{ext}'", file=sys.stderr)
        sys.exit(1)
    
    # Progress logging
    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)
    
    # Read file
    log(f"📄 Reading {input_path.name}...")
    file_bytes = input_path.read_bytes()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    log(f"   Size: {file_size_mb:.2f} MB")
    
    # Import Modal (do this late to avoid slow import on --help)
    try:
        import modal
    except ImportError:
        print("Error: Modal not installed. Run: pip install modal", file=sys.stderr)
        sys.exit(1)
    
    # Get the Modal function
    log("🔌 Connecting to Modal...")
    try:
        DolphinModel = modal.Cls.from_name("dolphin-pdf-parser", "DolphinModel")
        dolphin = DolphinModel()
    except modal.exception.NotFoundError:
        print("Error: Dolphin app not deployed.", file=sys.stderr)
        print("Run: cd modal && modal deploy dolphin_modal.py", file=sys.stderr)
        sys.exit(1)
    
    # Process the file
    log("🐬 Processing with Dolphin AI...")
    log("   (This may take up to 2 min per page for complex documents)")
    
    start_time = time.time()
    
    try:
        if ext == '.pdf':
            result = dolphin.parse_pdf.remote(file_bytes)
        else:
            result = dolphin.parse_image.remote(file_bytes)
    except Exception as e:
        print(f"Error: Processing failed - {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    elapsed = time.time() - start_time
    
    if not result.get("success"):
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        if result.get("traceback"):
            print(result["traceback"], file=sys.stderr)
        sys.exit(1)
    
    total_pages = result.get("total_pages", 1)
    log(f"✅ Processed {total_pages} page(s) in {elapsed:.1f}s")
    
    # Build structured output for LLMs
    structured_output = build_llm_json(result, input_path.name, elapsed)
    
    # Format output based on user preference
    if args.text_only:
        output = structured_output["full_text"]
    elif args.format == "markdown":
        output = result.get("markdown", "")
    elif args.format == "text":
        output = structured_output["full_text"]
    else:  # json
        indent = None if args.compact else 2
        output = json.dumps(structured_output, indent=indent, ensure_ascii=False)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output, encoding='utf-8')
        log(f"💾 Saved to {args.output}")
    else:
        print(output)


def build_llm_json(result: dict, filename: str, processing_time: float) -> dict:
    """
    Build a structured JSON format optimized for LLM consumption.
    
    Structure:
    {
        "metadata": {...},
        "pages": [...],
        "full_text": "...",
        "chapters": [...] (if detected)
    }
    """
    pages = result.get("pages", [])
    total_pages = result.get("total_pages", len(pages))
    
    # Build page-by-page structure
    structured_pages = []
    all_text_parts = []
    
    for page in pages:
        page_num = page.get("page_number", 0)
        elements = page.get("elements", [])
        page_markdown = page.get("markdown", "")
        
        # Extract text from elements
        page_text_parts = []
        structured_elements = []
        
        for elem in elements:
            label = elem.get("label", "text")
            text = elem.get("text", "").strip()
            
            if text:
                page_text_parts.append(text)
                
                # Map label to semantic type
                elem_type = map_label_to_type(label)
                
                structured_elements.append({
                    "type": elem_type,
                    "content": text,
                    "bbox": elem.get("bbox"),
                })
        
        page_text = "\n\n".join(page_text_parts)
        all_text_parts.append(page_text)
        
        structured_pages.append({
            "page_number": page_num,
            "elements": structured_elements,
            "text": page_text,
            "markdown": page_markdown,
        })
    
    # Combine all text
    full_text = "\n\n---\n\n".join(all_text_parts)
    
    # Try to detect chapters
    chapters = detect_chapters(structured_pages)
    
    # Build final structure
    output = {
        "metadata": {
            "source_file": filename,
            "total_pages": total_pages,
            "processing_time_seconds": round(processing_time, 2),
            "processed_at": datetime.now().isoformat(),
            "model": "ByteDance/Dolphin-v2",
        },
        "pages": structured_pages,
        "full_text": full_text,
    }
    
    if chapters:
        output["chapters"] = chapters
    
    return output


def map_label_to_type(label: str) -> str:
    """Map Dolphin element labels to semantic types."""
    label_map = {
        "title": "title",
        "sec_0": "heading",
        "sec_1": "heading",
        "sec_2": "subheading",
        "sec_3": "subheading",
        "paragraph": "paragraph",
        "text": "paragraph",
        "list_item": "list_item",
        "tab": "table",
        "table": "table",
        "equ": "equation",
        "formula": "equation",
        "code": "code",
        "fig": "figure",
        "figure": "figure",
        "caption": "caption",
        "fnote": "footnote",
        "distorted_page": "paragraph",
    }
    return label_map.get(label, "paragraph")


def detect_chapters(pages: list) -> list:
    """
    Try to detect chapter structure from the content.
    Returns a list of chapters with start/end pages.
    """
    chapters = []
    current_chapter = None
    
    chapter_keywords = ["chapter", "part", "section", "book"]
    
    for page in pages:
        page_num = page.get("page_number", 0)
        elements = page.get("elements", [])
        
        for elem in elements:
            if elem.get("type") in ["title", "heading"]:
                content = elem.get("content", "").lower()
                
                # Check if this looks like a chapter heading
                for keyword in chapter_keywords:
                    if keyword in content:
                        # Save previous chapter
                        if current_chapter:
                            current_chapter["end_page"] = page_num - 1
                            chapters.append(current_chapter)
                        
                        # Start new chapter
                        current_chapter = {
                            "title": elem.get("content", "").strip(),
                            "start_page": page_num,
                            "end_page": None,
                        }
                        break
    
    # Close last chapter
    if current_chapter:
        current_chapter["end_page"] = pages[-1].get("page_number", 0) if pages else 0
        chapters.append(current_chapter)
    
    return chapters


if __name__ == "__main__":
    main()
