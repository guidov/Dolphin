#!/usr/bin/env python3
"""
Storybook Parser - Convert children's picture books to text
============================================================
Optimized for illustrated storybooks - processes only text elements,
skipping tables, equations, and other complex elements.

Usage:
    python storybook_parser.py book.pdf                    # Output to stdout
    python storybook_parser.py book.pdf -o story.json      # Save as JSON
    python storybook_parser.py book.pdf -o story.txt --text-only  # Plain text

Examples:
    python storybook_parser.py "Wizard_of_Oz.pdf" -o wizard.json
    python storybook_parser.py "Charlotte's Web.pdf" --text-only > story.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Parse storybook PDFs - optimized for children's illustrated books",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("input", help="Input PDF file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Output only the extracted text (no JSON structure)"
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
    
    ext = input_path.suffix.lower()
    if ext != '.pdf':
        print(f"Error: Only PDF files are supported (got {ext})", file=sys.stderr)
        sys.exit(1)
    
    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)
    
    # Read file
    log(f"📚 Reading {input_path.name}...")
    file_bytes = input_path.read_bytes()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    log(f"   Size: {file_size_mb:.2f} MB")
    
    # Import Modal
    try:
        from modal import Cls
    except ImportError:
        print("Error: Modal not installed. Run: pip install modal", file=sys.stderr)
        sys.exit(1)
    
    # Connect to Modal
    log("🔌 Connecting to Modal...")
    try:
        DolphinModel = Cls.from_name("dolphin-pdf-parser", "DolphinModel")
        dolphin = DolphinModel()
    except Exception as e:
        print(f"Error: Could not connect to Modal - {e}", file=sys.stderr)
        print("Run: cd dolphin_modal_app && modal deploy dolphin_modal.py", file=sys.stderr)
        sys.exit(1)
    
    # Process storybook
    log("📖 Processing storybook (text-only mode)...")
    log("   (This skips tables, equations, and images for speed)")
    
    start_time = time.time()
    
    try:
        result = dolphin.parse_storybook.remote(file_bytes)
    except Exception as e:
        print(f"Error: Processing failed - {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    elapsed = time.time() - start_time
    
    if not result.get("success"):
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        if result.get("traceback"):
            print(result["traceback"], file=sys.stderr)
        sys.exit(1)
    
    total_pages = result.get("total_pages", 0)
    log(f"✅ Processed {total_pages} page(s) in {elapsed:.1f}s ({elapsed/total_pages:.1f}s/page)")
    
    # Format output
    if args.text_only:
        output = result.get("full_text", "")
    else:
        # Build structured JSON
        structured = {
            "metadata": {
                "source_file": input_path.name,
                "total_pages": total_pages,
                "processing_time_seconds": round(elapsed, 2),
                "processed_at": datetime.now().isoformat(),
                "mode": "storybook",
            },
            "pages": result.get("pages", []),
            "full_text": result.get("full_text", ""),
        }
        output = json.dumps(structured, indent=2, ensure_ascii=False)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output, encoding='utf-8')
        log(f"💾 Saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
