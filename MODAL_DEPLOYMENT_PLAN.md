# 🐬 Dolphin on Modal - Deployment Plan

## Overview

This plan outlines how to deploy the ByteDance Dolphin document parsing model on Modal cloud infrastructure with a web application front-end for PDF-to-text conversion.

---

## 🎯 Project Goals

1. **Deploy Dolphin model on Modal** - Run the 3B parameter Dolphin-v2 model on Modal's GPU infrastructure
2. **Create web endpoint API** - Accept PDF uploads and return extracted text/markdown
3. **Build a beautiful web app** - User-friendly interface for uploading PDFs and viewing results

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WEB APPLICATION                        │
│           (HTML/CSS/JS - served by Modal)                   │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Upload PDF │ -> │  Processing  │ -> │   Display    │   │
│  │   Dropzone  │    │   Status     │    │   Results    │   │
│  └─────────────┘    └──────────────┘    └──────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP POST /convert
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODAL BACKEND                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │             FastAPI Web Endpoint                     │   │
│  │     @modal.fastapi_endpoint()                        │   │
│  │     - Receives PDF uploads                           │   │
│  │     - Returns JSON/Markdown text                     │   │
│  └───────────────────────────┬─────────────────────────┘   │
│                              │                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Dolphin Processing Function                 │   │
│  │     @app.function(gpu="A100" or "L40S")              │   │
│  │     - Loads Dolphin-v2 model                         │   │
│  │     - Processes PDF pages                            │   │
│  │     - Returns structured text                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Modal Volume                           │   │
│  │     - Caches HuggingFace model weights               │   │
│  │     - Temporary file storage                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Implementation Phases

### Phase 1: Modal Backend Setup (Core)

#### 1.1 Create Modal Configuration File
**File:** `modal/dolphin_modal.py`

```python
# Key components:
import modal

app = modal.App("dolphin-pdf-parser")

# Define the container image with all dependencies
dolphin_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV dependencies
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0", 
        "transformers==4.51.0",
        "qwen_vl_utils",
        "pymupdf==1.26",
        "opencv-python",
        "Pillow",
        "numpy",
        "fastapi[standard]",
    )
    .add_local_python_source("utils")  # Add utils module
)

# Volume for caching model weights
model_volume = modal.Volume.from_name("dolphin-model-cache", create_if_missing=True)
```

#### 1.2 Create Model Loading Class
- Download and cache model from HuggingFace
- Use Modal's `run_function` or class-based approach for warm starts
- Implement `@modal.cls()` for persistent model in memory

```python
@app.cls(
    image=dolphin_image,
    gpu="A100-40GB",  # Or L40S for cost optimization
    volumes={"/model_cache": model_volume},
    container_idle_timeout=300,  # Keep warm for 5 min
)
class DolphinModel:
    @modal.enter()
    def load_model(self):
        # Load model into GPU on container start
        self.model = DOLPHIN("ByteDance/Dolphin-v2")
    
    @modal.method()
    def parse_pdf(self, pdf_bytes: bytes) -> dict:
        # Process PDF and return results
        ...
```

#### 1.3 Create Web Endpoint
```python
@app.function(image=dolphin_image)
@modal.fastapi_endpoint()
async def convert_pdf(file: UploadFile = File(...)):
    # Read uploaded PDF
    # Call DolphinModel.parse_pdf()
    # Return markdown/JSON results
```

---

### Phase 2: Web Application Frontend

#### 2.1 Static Web App Structure
**Files to create:**
- `modal/web/index.html` - Main page
- `modal/web/styles.css` - Styling
- `modal/web/app.js` - JavaScript logic

#### 2.2 Features
- **Drag-and-drop PDF upload** with visual feedback
- **Multi-page PDF support** with page navigation
- **Real-time processing status** indicator
- **Split-view display**: Original PDF | Extracted Text/Markdown
- **Download options**: JSON, Markdown, Plain Text
- **Dark/Light mode** toggle

#### 2.3 UI Design
```
┌──────────────────────────────────────────────────────────┐
│  🐬 Dolphin PDF Parser                      [Dark Mode]  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │                                                    │ │
│  │         📄 Drop your PDF here                     │ │
│  │            or click to browse                     │ │
│  │                                                    │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────────┬─────────────────────────────┐ │
│  │    PDF Preview       │     Extracted Content       │ │
│  │  ┌────────────────┐  │  ┌───────────────────────┐  │ │
│  │  │                │  │  │ # Document Title      │  │ │
│  │  │   [Page 1]     │  │  │                       │  │ │
│  │  │                │  │  │ Lorem ipsum dolor...  │  │ │
│  │  │                │  │  │                       │  │ │
│  │  └────────────────┘  │  │ | Table | Data |      │  │ │
│  │    [<] Page 1/5 [>]   │  │                       │  │ │
│  │                      │  └───────────────────────┘  │ │
│  └──────────────────────┴─────────────────────────────┘ │
│                                                          │
│  [Download Markdown] [Download JSON] [Copy to Clipboard] │
└──────────────────────────────────────────────────────────┘
```

---

### Phase 3: Integration & Deployment

#### 3.1 Serve Static Files via Modal
```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

web_app = FastAPI()
web_app.mount("/static", StaticFiles(directory="web"), name="static")

@app.function(image=dolphin_image)
@modal.asgi_app()
def serve_app():
    return web_app
```

#### 3.2 API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web application |
| `/api/convert` | POST | Upload and convert PDF |
| `/api/status/{job_id}` | GET | Check conversion status |
| `/api/download/{job_id}` | GET | Download results |

#### 3.3 Deployment Commands
```bash
# Install Modal CLI
pip install modal

# Login to Modal
modal token new

# Deploy the application
modal deploy modal/dolphin_modal.py

# Local development/testing
modal serve modal/dolphin_modal.py
```

---

## 📁 File Structure

```
Dolphin/
├── modal/
│   ├── __init__.py
│   ├── dolphin_modal.py      # Main Modal app
│   ├── dolphin_service.py    # Dolphin model wrapper
│   └── web/
│       ├── index.html        # Web UI
│       ├── styles.css        # Styling
│       └── app.js            # JavaScript
├── utils/
│   ├── utils.py              # Existing utilities
│   └── markdown_utils.py     # Markdown converter
├── requirements.txt          # Dependencies
└── MODAL_DEPLOYMENT_PLAN.md  # This file
```

---

## 💰 Cost Considerations

### GPU Options
| GPU | VRAM | Cost/hr | Recommended For |
|-----|------|---------|-----------------|
| L40S | 48GB | ~$1.00 | Production (cost-effective) |
| A100-40GB | 40GB | ~$1.79 | Production (reliable) |
| A100-80GB | 80GB | ~$2.49 | Large batch processing |
| H100 | 80GB | ~$3.95 | Maximum speed |

### Optimization Tips
1. **Container warm-up**: Use `container_idle_timeout` to keep model loaded
2. **Volume caching**: Cache model weights in Modal Volume
3. **Batch processing**: Process multiple pages concurrently where possible

---

## 🔧 Technical Requirements

### Modal Requirements
- Modal account with GPU access
- `modal` Python package installed
- Sufficient credits for GPU usage

### Model Requirements
- **Model**: ByteDance/Dolphin-v2 (~6GB)
- **Min VRAM**: 16GB (A10+)
- **Recommended VRAM**: 40GB (A100-40GB)

### Dependencies
All dependencies from `requirements.txt` plus:
- `fastapi[standard]` for web endpoints
- `python-multipart` for file uploads

---

## 🚀 Next Steps

1. [ ] **Phase 1.1**: Create `modal/dolphin_modal.py` with basic structure
2. [ ] **Phase 1.2**: Implement model loading and caching
3. [ ] **Phase 1.3**: Create PDF processing endpoint
4. [ ] **Phase 2.1**: Build web UI (HTML/CSS/JS)
5. [ ] **Phase 2.2**: Integrate frontend with backend
6. [ ] **Phase 3.1**: Test locally with `modal serve`
7. [ ] **Phase 3.2**: Deploy with `modal deploy`
8. [ ] **Phase 3.3**: Set up monitoring and error handling

---

## ⚠️ Potential Challenges

1. **Cold start latency**: First request after idle may take 30-60 seconds
   - *Solution*: Use `container_idle_timeout` and consider Modal's warm pools

2. **Large model size**: 6GB+ model weights
   - *Solution*: Use Modal Volume for persistent caching

3. **Multi-page PDFs**: Can be slow for large documents
   - *Solution*: Show progress updates, consider async processing

4. **Memory management**: Multiple pages in memory
   - *Solution*: Process pages sequentially, clear memory between pages

---

## 📚 References

- [Modal Documentation](https://modal.com/docs)
- [Modal GPU Guide](https://modal.com/docs/guide/gpu)
- [Modal Web Endpoints](https://modal.com/docs/guide/webhooks)
- [Dolphin Repository](https://github.com/bytedance/Dolphin)
- [Dolphin HuggingFace Model](https://huggingface.co/ByteDance/Dolphin-v2)

---

*Plan created: January 6, 2026*
