"""
Dolphin PDF Parser - Modal Cloud Deployment
============================================
Deploy ByteDance's Dolphin-v2 document parsing model on Modal's GPU infrastructure.

Usage:
    modal serve modal/dolphin_modal.py   # Local development
    modal deploy modal/dolphin_modal.py  # Production deployment
"""

import io
import os
import json
import tempfile
from pathlib import Path
from typing import Optional

import modal

# ============================================================================
# Modal Configuration
# ============================================================================

app = modal.App("dolphin-pdf-parser")

# Create a volume for caching model weights
model_volume = modal.Volume.from_name("dolphin-model-cache", create_if_missing=True)

# Define the container image with all dependencies
dolphin_image = (
    modal.Image.debian_slim(python_version="3.11")
    # System dependencies for OpenCV and image processing
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "git-lfs",
    )
    # Python dependencies
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.51.0",
        "accelerate==1.4.0",
        "qwen_vl_utils",
        "pymupdf==1.26",
        "opencv-python-headless",  # Headless for server
        "Pillow",
        "numpy",
        "fastapi[standard]",
        "python-multipart",
        "huggingface_hub",
    )
    # Add local utility modules
    .add_local_dir(
        str(Path(__file__).parent.parent / "utils"),
        remote_path="/root/utils",
        copy=True,
    )
    # Set up Python path
    .env({"PYTHONPATH": "/root"})
)

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_ID = "ByteDance/Dolphin-v2"
MODEL_CACHE_PATH = "/model_cache/dolphin-v2"


def download_model():
    """Download model weights to the cache volume."""
    from huggingface_hub import snapshot_download
    
    print(f"Downloading model {MODEL_ID} to {MODEL_CACHE_PATH}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_CACHE_PATH,
        local_dir_use_symlinks=False,
    )
    print("Model download complete!")


# ============================================================================
# Dolphin Model Class (GPU-accelerated)
# ============================================================================

@app.cls(
    image=dolphin_image,
    gpu="T4",  # Cheaper option (~$0.15/hr) - 16GB VRAM
    volumes={"/model_cache": model_volume},
    scaledown_window=300,  # Keep warm for 5 minutes
    timeout=36000,  # 10 hours for very large books
)
class DolphinModel:
    """
    Dolphin document parsing model running on Modal GPU infrastructure.
    Uses class-based approach for persistent model loading across requests.
    """
    
    @modal.enter()
    def load_model(self):
        """Load model into GPU memory when container starts."""
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        
        print("Loading Dolphin model into GPU memory...")
        
        # Check if model is cached
        if not os.path.exists(MODEL_CACHE_PATH):
            print("Model not in cache, downloading...")
            download_model()
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(MODEL_CACHE_PATH)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_CACHE_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        
        # Set up tokenizer
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        
        print("Model loaded successfully!")
    
    def _resize_image(self, image, max_size=1600, min_size=28):
        """Resize image to appropriate dimensions."""
        width, height = image.size
        if max(width, height) < max_size and min(width, height) >= 28:
            return image
        
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            image = image.resize((new_width, new_height))
            width, height = image.size
        
        if min(width, height) < 28:
            if width < height:
                new_width = min_size
                new_height = int(height * (min_size / width))
            else:
                new_height = min_size
                new_width = int(width * (min_size / height))
            image = image.resize((new_width, new_height))
        
        return image
    
    def _chat(self, prompt, image):
        """
        Run inference on the model. Supports both single and batch inputs.
        
        Args:
            prompt: Single prompt string or list of prompts
            image: Single PIL Image or list of PIL Images
            
        Returns:
            Single result string or list of result strings
        """
        from qwen_vl_utils import process_vision_info
        
        # Check if we're dealing with a batch
        is_batch = isinstance(image, list)
        
        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        
        assert len(images) == len(prompts)
        
        # Preprocess all images
        processed_images = [self._resize_image(img) for img in images]
        
        # Generate all messages
        all_messages = []
        for img, question in zip(processed_images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": question}
                    ],
                }
            ]
            all_messages.append(messages)
        
        # Prepare all texts
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in all_messages
        ]
        
        # Collect all image inputs
        all_image_inputs = []
        for msgs in all_messages:
            image_inputs, _ = process_vision_info(msgs)
            all_image_inputs.extend(image_inputs)
        
        # Prepare model inputs
        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        results = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Return single result for single input
        if not is_batch:
            return results[0]
        return results
    
    @modal.method()
    def parse_pdf(self, pdf_bytes: bytes) -> dict:
        """
        Parse a PDF document and extract text content.
        Uses two-stage parsing: layout analysis + element-level recognition.
        
        Args:
            pdf_bytes: Raw PDF file bytes
            
        Returns:
            Dictionary containing:
                - success: bool
                - pages: list of page results
                - markdown: combined markdown text
                - total_pages: int
        """
        import pymupdf
        from PIL import Image
        from utils.utils import parse_layout_string, process_coordinates
        from utils.markdown_utils import MarkdownConverter
        
        try:
            # Open PDF from bytes
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            
            all_page_results = []
            all_markdown_parts = []
            markdown_converter = MarkdownConverter()
            
            for page_idx in range(total_pages):
                print(f"Processing page {page_idx + 1}/{total_pages}...")
                
                page = doc[page_idx]
                
                # Render page as image
                target_size = 896
                rect = page.rect
                scale = target_size / max(rect.width, rect.height)
                mat = pymupdf.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Stage 1: Layout analysis - get reading order and element bboxes
                layout_output = self._chat(
                    "Parse the reading order of this document.",
                    pil_image
                )
                
                # Parse layout to get element list
                layout_list = parse_layout_string(layout_output)
                
                # Handle edge cases
                if not layout_list or not (layout_output.startswith("[") and layout_output.endswith("]")):
                    # Treat entire page as one text element
                    layout_list = [([0, 0, pil_image.size[0], pil_image.size[1]], 'distorted_page', [])]
                
                # Stage 2: Batched element-level recognition
                # Group elements by type for batch processing
                MAX_BATCH_SIZE = 4  # Process 4 elements at a time
                
                text_elements = []  # Regular text
                tab_elements = []   # Tables
                equ_elements = []   # Equations
                code_elements = []  # Code blocks
                figure_results = [] # Figures (no OCR needed)
                
                reading_order = 0
                
                for bbox, label, tags in layout_list:
                    try:
                        # Get coordinates
                        if label == "distorted_page":
                            x1, y1, x2, y2 = 0, 0, pil_image.size[0], pil_image.size[1]
                            crop = pil_image
                        else:
                            x1, y1, x2, y2 = process_coordinates(bbox, pil_image)
                            crop = pil_image.crop((x1, y1, x2, y2))
                        
                        # Skip tiny crops
                        if crop.size[0] <= 3 or crop.size[1] <= 3:
                            continue
                        
                        element_info = {
                            "crop": crop,
                            "label": label,
                            "bbox": [x1, y1, x2, y2],
                            "reading_order": reading_order,
                            "tags": tags,
                        }
                        
                        # Group by element type
                        if label == "fig":
                            figure_results.append({
                                "label": label,
                                "text": "[Figure]",
                                "bbox": [x1, y1, x2, y2],
                                "reading_order": reading_order,
                            })
                        elif label == "tab":
                            tab_elements.append(element_info)
                        elif label == "equ":
                            equ_elements.append(element_info)
                        elif label == "code":
                            code_elements.append(element_info)
                        else:
                            text_elements.append(element_info)
                        
                        reading_order += 1
                        
                    except Exception as e:
                        print(f"Error preparing element: {e}")
                        continue
                
                # Helper function to process elements in batches
                def process_batch(elements, prompt):
                    results = []
                    for i in range(0, len(elements), MAX_BATCH_SIZE):
                        batch = elements[i:i + MAX_BATCH_SIZE]
                        crops = [e["crop"] for e in batch]
                        prompts = [prompt] * len(crops)
                        
                        # Batch inference
                        batch_results = self._chat(prompts, crops)
                        
                        for j, text in enumerate(batch_results):
                            elem = batch[j]
                            results.append({
                                "label": elem["label"],
                                "text": text.strip(),
                                "bbox": elem["bbox"],
                                "reading_order": elem["reading_order"],
                            })
                    return results
                
                # Process each element type in batches
                recognition_results = figure_results.copy()
                
                if tab_elements:
                    print(f"  Processing {len(tab_elements)} tables in batches...")
                    recognition_results.extend(
                        process_batch(tab_elements, "Convert the table to markdown format.")
                    )
                
                if equ_elements:
                    print(f"  Processing {len(equ_elements)} equations in batches...")
                    recognition_results.extend(
                        process_batch(equ_elements, "Convert the equation to LaTeX format.")
                    )
                
                if code_elements:
                    print(f"  Processing {len(code_elements)} code blocks in batches...")
                    recognition_results.extend(
                        process_batch(code_elements, "Read the code in the image.")
                    )
                
                if text_elements:
                    print(f"  Processing {len(text_elements)} text elements in batches...")
                    recognition_results.extend(
                        process_batch(text_elements, "Read text in the image.")
                    )
                
                # Sort by reading order
                recognition_results.sort(key=lambda x: x.get("reading_order", 0))
                
                # Convert to markdown
                try:
                    page_markdown = markdown_converter.convert(recognition_results)
                except Exception:
                    # Fallback: just join all text
                    page_markdown = "\n\n".join([r.get("text", "") for r in recognition_results])
                
                page_result = {
                    "page_number": page_idx + 1,
                    "elements": recognition_results,
                    "markdown": page_markdown,
                }
                all_page_results.append(page_result)
                all_markdown_parts.append(f"## Page {page_idx + 1}\n\n{page_markdown}")
            
            doc.close()
            
            # Combine all markdown
            combined_markdown = "\n\n---\n\n".join(all_markdown_parts)
            
            return {
                "success": True,
                "total_pages": total_pages,
                "pages": all_page_results,
                "markdown": combined_markdown,
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "total_pages": 0,
                "pages": [],
                "markdown": "",
            }
    
    @modal.method()
    def parse_storybook(self, pdf_bytes: bytes) -> dict:
        """
        Parse a storybook PDF - optimized for children's books.
        Only processes TEXT elements (no tables, equations, code).
        Much faster than parse_pdf for simple illustrated books.
        
        Args:
            pdf_bytes: Raw PDF file bytes
            
        Returns:
            Dictionary with extracted text content
        """
        import pymupdf
        from PIL import Image
        from utils.utils import parse_layout_string, process_coordinates
        
        MAX_BATCH_SIZE = 4
        
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            
            all_page_results = []
            all_text_parts = []
            
            for page_idx in range(total_pages):
                print(f"Processing page {page_idx + 1}/{total_pages}...")
                
                page = doc[page_idx]
                
                # Render page as image
                target_size = 896
                rect = page.rect
                scale = target_size / max(rect.width, rect.height)
                mat = pymupdf.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat)
                
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Layout analysis
                layout_output = self._chat(
                    "Parse the reading order of this document.",
                    pil_image
                )
                
                layout_list = parse_layout_string(layout_output)
                
                if not layout_list or not (layout_output.startswith("[") and layout_output.endswith("]")):
                    layout_list = [([0, 0, pil_image.size[0], pil_image.size[1]], 'distorted_page', [])]
                
                # Collect ONLY text elements (skip tables, equations, code, figures)
                text_elements = []
                reading_order = 0
                
                for bbox, label, tags in layout_list:
                    try:
                        # Skip non-text elements
                        if label in ["tab", "equ", "code", "fig", "figure", "table"]:
                            reading_order += 1
                            continue
                        
                        if label == "distorted_page":
                            x1, y1, x2, y2 = 0, 0, pil_image.size[0], pil_image.size[1]
                            crop = pil_image
                        else:
                            x1, y1, x2, y2 = process_coordinates(bbox, pil_image)
                            crop = pil_image.crop((x1, y1, x2, y2))
                        
                        if crop.size[0] <= 3 or crop.size[1] <= 3:
                            continue
                        
                        text_elements.append({
                            "crop": crop,
                            "label": label,
                            "bbox": [x1, y1, x2, y2],
                            "reading_order": reading_order,
                        })
                        reading_order += 1
                        
                    except Exception as e:
                        print(f"Error preparing element: {e}")
                        continue
                
                # Batch process text elements
                recognition_results = []
                
                if text_elements:
                    print(f"  Processing {len(text_elements)} text elements...")
                    
                    for i in range(0, len(text_elements), MAX_BATCH_SIZE):
                        batch = text_elements[i:i + MAX_BATCH_SIZE]
                        crops = [e["crop"] for e in batch]
                        prompts = ["Read text in the image."] * len(crops)
                        
                        batch_results = self._chat(prompts, crops)
                        
                        for j, text in enumerate(batch_results):
                            elem = batch[j]
                            recognition_results.append({
                                "label": elem["label"],
                                "text": text.strip(),
                                "bbox": elem["bbox"],
                                "reading_order": elem["reading_order"],
                            })
                
                # Sort and combine text
                recognition_results.sort(key=lambda x: x.get("reading_order", 0))
                page_text = "\n\n".join([r.get("text", "") for r in recognition_results])
                
                page_result = {
                    "page_number": page_idx + 1,
                    "elements": recognition_results,
                    "text": page_text,
                }
                all_page_results.append(page_result)
                
                if page_text.strip():
                    all_text_parts.append(f"--- Page {page_idx + 1} ---\n\n{page_text}")
            
            doc.close()
            
            full_text = "\n\n".join(all_text_parts)
            
            return {
                "success": True,
                "total_pages": total_pages,
                "pages": all_page_results,
                "full_text": full_text,
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "total_pages": 0,
                "pages": [],
                "full_text": "",
            }
    
    @modal.method()
    def parse_image(self, image_bytes: bytes) -> dict:
        """
        Parse a single image and extract text content.
        Uses two-stage parsing: layout analysis + element-level recognition.
        
        Args:
            image_bytes: Raw image file bytes
            
        Returns:
            Dictionary with extracted content
        """
        from PIL import Image
        from utils.utils import parse_layout_string, process_coordinates
        from utils.markdown_utils import MarkdownConverter
        
        try:
            # Open image from bytes
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Stage 1: Layout analysis
            layout_output = self._chat(
                "Parse the reading order of this document.",
                pil_image
            )
            
            # Parse layout
            layout_list = parse_layout_string(layout_output)
            
            if not layout_list or not (layout_output.startswith("[") and layout_output.endswith("]")):
                layout_list = [([0, 0, pil_image.size[0], pil_image.size[1]], 'distorted_page', [])]
            
            # Stage 2: Element-level recognition
            recognition_results = []
            reading_order = 0
            
            for bbox, label, tags in layout_list:
                try:
                    if label == "distorted_page":
                        x1, y1, x2, y2 = 0, 0, pil_image.size[0], pil_image.size[1]
                        crop = pil_image
                    else:
                        x1, y1, x2, y2 = process_coordinates(bbox, pil_image)
                        crop = pil_image.crop((x1, y1, x2, y2))
                    
                    if crop.size[0] <= 3 or crop.size[1] <= 3:
                        continue
                    
                    if label == "fig":
                        recognition_results.append({
                            "label": label,
                            "text": "[Figure]",
                            "bbox": [x1, y1, x2, y2],
                            "reading_order": reading_order,
                        })
                        reading_order += 1
                        continue
                    
                    if label == "tab":
                        prompt = "Convert the table to markdown format."
                    elif label == "equ":
                        prompt = "Convert the equation to LaTeX format."
                    elif label == "code":
                        prompt = "Read the code in the image."
                    else:
                        prompt = "Read text in the image."
                    
                    text_output = self._chat(prompt, crop)
                    
                    recognition_results.append({
                        "label": label,
                        "text": text_output.strip(),
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                    })
                    reading_order += 1
                    
                except Exception as e:
                    print(f"Error processing element: {e}")
                    continue
            
            # Sort and convert to markdown
            recognition_results.sort(key=lambda x: x.get("reading_order", 0))
            
            try:
                markdown_converter = MarkdownConverter()
                markdown_output = markdown_converter.convert(recognition_results)
            except Exception:
                markdown_output = "\n\n".join([r.get("text", "") for r in recognition_results])
            
            return {
                "success": True,
                "elements": recognition_results,
                "markdown": markdown_output,
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "markdown": "",
            }


# ============================================================================
# Async OCR Processing (for background jobs)
# ============================================================================

# Image with boto3 for R2 access and supabase client
async_processing_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "boto3",
        "supabase",
    )
)

@app.function(
    image=async_processing_image,
    timeout=36000,  # 10 hours
    secrets=[
        modal.Secret.from_name("r2-credentials"),
        modal.Secret.from_name("supabase-credentials"),
    ],
)
def process_book_ocr_async(book_id: str, file_key: str, book_title: str):
    """
    Async OCR processing that runs independently of HTTP connections.
    
    This function is designed to be called with .spawn() for fire-and-forget execution.
    It handles:
    1. Fetching PDF from R2
    2. Calling Dolphin OCR
    3. Saving result to R2
    4. Updating Supabase cache status
    
    Requires Modal secrets:
    - r2-credentials: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
    - supabase-credentials: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    """
    import boto3
    from supabase import create_client
    import time
    
    start_time = time.time()
    
    # Initialize clients from secrets
    r2_account_id = os.environ["R2_ACCOUNT_ID"]
    r2_access_key = os.environ["R2_ACCESS_KEY_ID"]
    r2_secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    r2_bucket = os.environ["R2_BUCKET_NAME"]
    
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    
    s3 = boto3.client(
        's3',
        endpoint_url=f'https://{r2_account_id}.r2.cloudflarestorage.com',
        aws_access_key_id=r2_access_key,
        aws_secret_access_key=r2_secret_key,
    )
    
    supabase = create_client(supabase_url, supabase_key)
    
    try:
        print(f"[Async] Starting OCR for book {book_id}")
        
        # 1. Fetch PDF from R2
        response = s3.get_object(Bucket=r2_bucket, Key=file_key)
        pdf_bytes = response['Body'].read()
        print(f"[Async] PDF fetched: {len(pdf_bytes)} bytes")
        
        # 2. Process with Dolphin
        dolphin = DolphinModel()
        result = dolphin.parse_storybook.remote(pdf_bytes)
        
        if not result.get('success'):
            raise Exception(result.get('error', 'OCR processing failed'))
        
        print(f"[Async] OCR completed: {result['total_pages']} pages")
        
        # 3. Add metadata and clean text
        processing_time = time.time() - start_time
        enriched_result = {
            **result,
            "metadata": {
                "book_id": book_id,
                "book_title": book_title,
                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "processing_time_ms": int(processing_time * 1000),
            },
            "clean_text": clean_ocr_text_simple(result.get('full_text', '')),
        }
        
        # 4. Save to R2
        json_key = f"ocr-json/{book_id}.json"
        s3.put_object(
            Bucket=r2_bucket,
            Key=json_key,
            Body=json.dumps(enriched_result),
            ContentType='application/json',
        )
        print(f"[Async] Saved OCR JSON to R2: {json_key}")
        
        # 5. Update Supabase
        supabase.table('book_ocr_cache').update({
            'status': 'completed',
            'page_count': result['total_pages'],
            'processing_time_seconds': processing_time,
            'error_message': None,
        }).eq('book_id', book_id).execute()
        
        print(f"[Async] OCR completed for book {book_id} in {processing_time:.1f}s")
        return {"success": True, "pages": result['total_pages']}
        
    except Exception as e:
        print(f"[Async] Error processing book {book_id}: {e}")
        # Update Supabase with error
        try:
            supabase.table('book_ocr_cache').update({
                'status': 'failed',
                'error_message': str(e),
            }).eq('book_id', book_id).execute()
        except:
            pass
        return {"success": False, "error": str(e)}


def clean_ocr_text_simple(text: str) -> str:
    """Simple text cleanup for OCR output."""
    import re
    if not text:
        return ''
    
    cleaned = text
    cleaned = re.sub(r'--- Page \d+ ---\n*', '', cleaned)
    cleaned = re.sub(r'^\d+\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'(\w)-\n(\w)', r'\1\2', cleaned)
    cleaned = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned)
    cleaned = re.sub(r'\n{2,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
    paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip() and len(p.strip()) > 3]
    return '\n\n'.join(paragraphs)


# ============================================================================
# Web Application
# ============================================================================

# Create the web frontend image (lighter, no GPU needed)
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]", "python-multipart")
)

# Load static files
static_path = Path(__file__).parent / "web"



@app.function(image=web_image, timeout=36000)  # 10 hours for large books
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web_app():
    """Serve the web application and API endpoints."""
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    
    api = FastAPI(title="Dolphin PDF Parser")
    
    # Get reference to the Dolphin model
    dolphin = DolphinModel()
    
    @api.get("/", response_class=HTMLResponse)
    async def home():
        """Serve the main web page."""
        html_file = static_path / "index.html"
        if html_file.exists():
            return HTMLResponse(content=html_file.read_text())
        else:
            # Return embedded HTML if file doesn't exist
            return HTMLResponse(content=get_embedded_html())
    
    @api.get("/styles.css")
    async def styles():
        """Serve CSS file."""
        css_file = static_path / "styles.css"
        if css_file.exists():
            return Response(content=css_file.read_text(), media_type="text/css")
        return Response(content=get_embedded_css(), media_type="text/css")
    
    @api.get("/app.js")
    async def scripts():
        """Serve JavaScript file."""
        js_file = static_path / "app.js"
        if js_file.exists():
            return Response(content=js_file.read_text(), media_type="application/javascript")
        return Response(content=get_embedded_js(), media_type="application/javascript")
    
    @api.post("/api/convert")
    async def convert_document(file: UploadFile = File(...)):
        """
        Convert a PDF or image to text/markdown.
        
        Accepts: PDF, PNG, JPG, JPEG files
        Returns: JSON with markdown content and page details
        """
        # Validate file type
        filename = file.filename.lower()
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg'}
        ext = '.' + filename.split('.')[-1] if '.' in filename else ''
        
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        try:
            if ext == '.pdf':
                # Process PDF
                result = dolphin.parse_pdf.remote(content)
            else:
                # Process image
                result = dolphin.parse_image.remote(content)
            
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Processing error: {str(e)}"
            )
    
    @api.post("/api/convert/storybook")
    async def convert_storybook(file: UploadFile = File(...)):
        """
        Convert a storybook PDF to text - optimized for children's books.
        
        This endpoint is faster than /api/convert because it:
        - Only processes TEXT elements (no tables, equations, code)
        - Optimized for illustrated storybooks
        
        Accepts: PDF files only
        Returns: JSON with extracted text and page details
        """
        # Validate file type
        filename = file.filename.lower() if file.filename else 'document.pdf'
        ext = '.' + filename.split('.')[-1] if '.' in filename else ''
        
        if ext != '.pdf':
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported for storybook conversion"
            )
        
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )
        
        try:
            # Process storybook PDF
            result = dolphin.parse_storybook.remote(content)
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Processing error: {str(e)}"
            )
    
    @api.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "model": MODEL_ID}
    
    @api.post("/api/start-ocr")
    async def start_ocr_async(
        book_id: str,
        file_key: str,
        book_title: str = "Untitled"
    ):
        """
        Start async OCR processing that runs independently of this HTTP connection.
        
        This immediately returns 202 Accepted and processes in the background.
        The result will be saved to R2 and Supabase when complete.
        """
        try:
            # Spawn the async processing function (fire-and-forget)
            # This returns immediately and processing continues in background
            process_book_ocr_async.spawn(book_id, file_key, book_title)
            
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "message": f"OCR started for book {book_id}. Processing will continue in background.",
                    "book_id": book_id,
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start OCR: {str(e)}"
            )
    
    return api


# ============================================================================
# Embedded Web Assets (Fallback)
# ============================================================================

def get_embedded_html():
    """Return embedded HTML for the web app."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐬 Dolphin PDF Parser</title>
    <link rel="stylesheet" href="/styles.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="app-container">
        <header class="header">
            <div class="logo">
                <span class="logo-icon">🐬</span>
                <h1>Dolphin PDF Parser</h1>
            </div>
            <p class="tagline">AI-powered document parsing with ByteDance's Dolphin-v2</p>
        </header>

        <main class="main-content">
            <!-- Upload Section -->
            <div id="upload-section" class="upload-section">
                <div class="upload-zone" id="drop-zone">
                    <div class="upload-icon">📄</div>
                    <h2>Drop your PDF here</h2>
                    <p>or click to browse</p>
                    <input type="file" id="file-input" accept=".pdf,.png,.jpg,.jpeg" hidden>
                    <button class="browse-btn" onclick="document.getElementById('file-input').click()">
                        Browse Files
                    </button>
                    <p class="file-types">Supports: PDF, PNG, JPG, JPEG</p>
                </div>
            </div>

            <!-- Processing Section -->
            <div id="processing-section" class="processing-section hidden">
                <div class="processing-spinner"></div>
                <h2>Processing your document...</h2>
                <p id="processing-status">Uploading file...</p>
                <p class="processing-note">This may take 30-60 seconds for the first request</p>
            </div>

            <!-- Results Section -->
            <div id="results-section" class="results-section hidden">
                <div class="results-header">
                    <h2>📝 Extracted Content</h2>
                    <div class="results-actions">
                        <button class="action-btn" onclick="copyToClipboard()">
                            📋 Copy
                        </button>
                        <button class="action-btn" onclick="downloadMarkdown()">
                            ⬇️ Download MD
                        </button>
                        <button class="action-btn secondary" onclick="resetApp()">
                            🔄 New Document
                        </button>
                    </div>
                </div>
                
                <div class="results-container">
                    <div class="results-content" id="results-content">
                        <!-- Content will be inserted here -->
                    </div>
                </div>
                
                <div class="page-info" id="page-info"></div>
            </div>

            <!-- Error Section -->
            <div id="error-section" class="error-section hidden">
                <div class="error-icon">❌</div>
                <h2>Something went wrong</h2>
                <p id="error-message"></p>
                <button class="action-btn" onclick="resetApp()">Try Again</button>
            </div>
        </main>

        <footer class="footer">
            <p>Powered by <a href="https://github.com/bytedance/Dolphin" target="_blank">ByteDance Dolphin-v2</a> 
               running on <a href="https://modal.com" target="_blank">Modal</a></p>
        </footer>
    </div>

    <script src="/app.js"></script>
</body>
</html>'''


def get_embedded_css():
    """Return embedded CSS for the web app."""
    return '''/* Dolphin PDF Parser - Styles */

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-tertiary: #1a1a25;
    --text-primary: #ffffff;
    --text-secondary: #a0a0b0;
    --accent-primary: #6366f1;
    --accent-secondary: #818cf8;
    --accent-glow: rgba(99, 102, 241, 0.3);
    --success: #10b981;
    --error: #ef4444;
    --border-color: rgba(255, 255, 255, 0.1);
    --gradient-start: #6366f1;
    --gradient-end: #06b6d4;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.6;
}

.app-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 3rem;
}

.logo {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}

.logo-icon {
    font-size: 2.5rem;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.logo h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.tagline {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

/* Main Content */
.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* Upload Section */
.upload-section {
    width: 100%;
    max-width: 600px;
}

.upload-zone {
    background: var(--bg-secondary);
    border: 2px dashed var(--border-color);
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}

.upload-zone:hover,
.upload-zone.drag-over {
    border-color: var(--accent-primary);
    background: var(--bg-tertiary);
    box-shadow: 0 0 40px var(--accent-glow);
}

.upload-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}

.upload-zone h2 {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.upload-zone p {
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}

.browse-btn {
    background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
    color: white;
    border: none;
    padding: 0.875rem 2rem;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.browse-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px var(--accent-glow);
}

.file-types {
    margin-top: 1.5rem;
    font-size: 0.875rem;
    color: var(--text-secondary);
}

/* Processing Section */
.processing-section {
    text-align: center;
    padding: 4rem 2rem;
}

.processing-spinner {
    width: 60px;
    height: 60px;
    border: 4px solid var(--bg-tertiary);
    border-top-color: var(--accent-primary);
    border-radius: 50%;
    margin: 0 auto 2rem;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.processing-section h2 {
    margin-bottom: 0.5rem;
}

.processing-section p {
    color: var(--text-secondary);
}

.processing-note {
    margin-top: 1rem;
    font-size: 0.875rem;
    opacity: 0.7;
}

/* Results Section */
.results-section {
    width: 100%;
}

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    gap: 1rem;
}

.results-header h2 {
    font-size: 1.5rem;
}

.results-actions {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}

.action-btn {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 0.625rem 1.25rem;
    border-radius: 8px;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.action-btn:hover {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
}

.action-btn.secondary {
    background: transparent;
}

.results-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    overflow: hidden;
}

.results-content {
    padding: 2rem;
    max-height: 600px;
    overflow-y: auto;
    font-size: 0.95rem;
    line-height: 1.8;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.results-content h1, .results-content h2, .results-content h3 {
    color: var(--accent-secondary);
    margin: 1.5rem 0 0.75rem;
}

.results-content h1:first-child,
.results-content h2:first-child {
    margin-top: 0;
}

.results-content code {
    background: var(--bg-tertiary);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: 'Fira Code', monospace;
}

.results-content pre {
    background: var(--bg-tertiary);
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
}

.results-content table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}

.results-content th, .results-content td {
    border: 1px solid var(--border-color);
    padding: 0.75rem;
    text-align: left;
}

.results-content th {
    background: var(--bg-tertiary);
}

.page-info {
    text-align: center;
    padding: 1rem;
    color: var(--text-secondary);
    font-size: 0.875rem;
}

/* Error Section */
.error-section {
    text-align: center;
    padding: 4rem 2rem;
}

.error-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}

.error-section h2 {
    margin-bottom: 0.5rem;
    color: var(--error);
}

.error-section p {
    color: var(--text-secondary);
    margin-bottom: 2rem;
    max-width: 400px;
    margin-left: auto;
    margin-right: auto;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    margin-top: auto;
    color: var(--text-secondary);
    font-size: 0.875rem;
}

.footer a {
    color: var(--accent-secondary);
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}

/* Utility */
.hidden {
    display: none !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-tertiary);
}

::-webkit-scrollbar-thumb {
    background: var(--accent-primary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-secondary);
}

/* Responsive */
@media (max-width: 768px) {
    .app-container {
        padding: 1rem;
    }
    
    .logo h1 {
        font-size: 1.5rem;
    }
    
    .upload-zone {
        padding: 3rem 1.5rem;
    }
    
    .results-header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .results-actions {
        width: 100%;
        justify-content: flex-start;
    }
}'''


def get_embedded_js():
    """Return embedded JavaScript for the web app."""
    return '''// Dolphin PDF Parser - Frontend JavaScript

// State
let currentResult = null;

// DOM Elements
const uploadSection = document.getElementById('upload-section');
const processingSection = document.getElementById('processing-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const processingStatus = document.getElementById('processing-status');
const resultsContent = document.getElementById('results-content');
const pageInfo = document.getElementById('page-info');
const errorMessage = document.getElementById('error-message');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
    setupFileInput();
});

// Drag and Drop
function setupDragAndDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
}

// File Input
function setupFileInput() {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

// Handle File Upload
async function handleFile(file) {
    // Validate file type
    const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
    const allowedExtensions = ['.pdf', '.png', '.jpg', '.jpeg'];
    
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(ext)) {
        showError(`Unsupported file type. Please upload PDF, PNG, or JPG files.`);
        return;
    }

    // Show processing
    showSection('processing');
    updateStatus('Uploading file...');

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', file);

        // Upload and process
        updateStatus('Processing document with AI...');
        
        const response = await fetch('/api/convert', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Processing failed');
        }

        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Processing failed');
        }

        // Store result and show
        currentResult = result;
        showResults(result, file.name);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An unexpected error occurred');
    }
}

// Update processing status
function updateStatus(status) {
    processingStatus.textContent = status;
}

// Show specific section
function showSection(section) {
    uploadSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');

    switch (section) {
        case 'upload':
            uploadSection.classList.remove('hidden');
            break;
        case 'processing':
            processingSection.classList.remove('hidden');
            break;
        case 'results':
            resultsSection.classList.remove('hidden');
            break;
        case 'error':
            errorSection.classList.remove('hidden');
            break;
    }
}

// Show results
function showResults(result, filename) {
    // Format content with basic markdown rendering
    const content = result.markdown || result.content || 'No content extracted';
    resultsContent.innerHTML = formatMarkdown(content);
    
    // Show page info
    if (result.total_pages) {
        pageInfo.textContent = `Extracted from ${result.total_pages} page(s) · ${filename}`;
    } else {
        pageInfo.textContent = `Extracted from ${filename}`;
    }
    
    showSection('results');
}

// Basic markdown formatting
function formatMarkdown(text) {
    // Escape HTML
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Headers
    html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    
    // Bold and italic
    html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    html = html.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
    
    // Code blocks
    html = html.replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Line breaks
    html = html.replace(/\\n/g, '<br>');
    
    return html;
}

// Show error
function showError(message) {
    errorMessage.textContent = message;
    showSection('error');
}

// Reset app
function resetApp() {
    currentResult = null;
    fileInput.value = '';
    showSection('upload');
}

// Copy to clipboard
async function copyToClipboard() {
    if (!currentResult) return;
    
    const text = currentResult.markdown || currentResult.content || '';
    
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!');
    } catch (err) {
        // Fallback
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showToast('Copied to clipboard!');
    }
}

// Download as markdown
function downloadMarkdown() {
    if (!currentResult) return;
    
    const text = currentResult.markdown || currentResult.content || '';
    const blob = new Blob([text], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'extracted-content.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Toast notification
function showToast(message) {
    // Create toast element
    const toast = document.createElement('div');
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        background: var(--accent-primary);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
        z-index: 1000;
        animation: fadeInOut 2s ease-in-out forwards;
    `;
    
    // Add animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeInOut {
            0% { opacity: 0; transform: translateX(-50%) translateY(20px); }
            15% { opacity: 1; transform: translateX(-50%) translateY(0); }
            85% { opacity: 1; transform: translateX(-50%) translateY(0); }
            100% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
        }
    `;
    document.head.appendChild(style);
    document.body.appendChild(toast);
    
    // Remove after animation
    setTimeout(() => {
        toast.remove();
        style.remove();
    }, 2000);
}'''


# ============================================================================
# CLI Entry Point for Local Testing
# ============================================================================

@app.local_entrypoint()
def main():
    """Test the Dolphin model locally."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: modal run modal/dolphin_modal.py <pdf_path>")
        print("Example: modal run modal/dolphin_modal.py ./demo/page_imgs/page_6.pdf")
        return
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    print(f"Processing: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    dolphin = DolphinModel()
    result = dolphin.parse_pdf.remote(pdf_bytes)
    
    if result["success"]:
        print(f"\n✅ Successfully processed {result['total_pages']} page(s)\n")
        print("=" * 60)
        print(result["markdown"])
        print("=" * 60)
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
