# 🐬 Dolphin Modal Deployment

Deploy ByteDance's Dolphin-v2 document parsing model on Modal cloud infrastructure.

## Quick Start

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal token new
```

This opens a browser to log in to your Modal account. Create a free account at [modal.com](https://modal.com) if you don't have one.

### 3. Deploy

#### Development Mode (with hot reload)

```bash
cd modal
modal serve dolphin_modal.py
```

This starts a local development server with live reload. You'll see a URL like:
```
→ Web app: https://your-name--dolphin-pdf-parser-web-app-dev.modal.run
```

#### Production Deployment

```bash
cd modal
modal deploy dolphin_modal.py
```

This deploys to Modal cloud permanently. You'll get a stable URL like:
```
→ Web app: https://your-name--dolphin-pdf-parser-web-app.modal.run
```

## Usage

### Web Interface

1. Open the deployed URL in your browser
2. Drag & drop a PDF or image file
3. Wait for AI processing (30-60 seconds on first request)
4. Download the extracted markdown/JSON

### API Usage

```bash
# Convert a PDF
curl -X POST "https://your-url.modal.run/api/convert" \
  -F "file=@document.pdf"

# Health check
curl "https://your-url.modal.run/api/health"
```

### Python Client

```python
import requests

# Upload and convert PDF
with open("document.pdf", "rb") as f:
    response = requests.post(
        "https://your-url.modal.run/api/convert",
        files={"file": f}
    )

result = response.json()
print(result["markdown"])
```

## Configuration

### GPU Selection

Edit `dolphin_modal.py` to change the GPU:

```python
@app.cls(
    gpu="L40S",  # Options: T4, L4, A10, A100, A100-40GB, L40S, H100
    ...
)
```

**Recommended GPUs:**
| GPU | VRAM | Cost/hr | Notes |
|-----|------|---------|-------|
| L40S | 48GB | ~$1.00 | Best cost/performance |
| A100-40GB | 40GB | ~$1.79 | Reliable, good performance |
| A100-80GB | 80GB | ~$2.49 | Large batches |

### Container Settings

```python
container_idle_timeout=300,  # Keep warm for 5 min (reduces cold starts)
timeout=600,                 # Max processing time per request
```

## Project Structure

```
modal/
├── __init__.py           # Package init
├── dolphin_modal.py      # Main Modal app
├── setup.sh              # Setup script
├── README.md             # This file
└── web/                  # Frontend assets
    ├── index.html
    ├── styles.css
    └── app.js
```

## Troubleshooting

### "Model download failed"

The model weights (~6GB) are cached in a Modal Volume. First deployment may take 5-10 minutes.

### "Out of memory"

Try a larger GPU (A100-80GB) or reduce `max_batch_size` in the code.

### "Cold start too slow"

- Increase `container_idle_timeout` to keep containers warm longer
- Use Modal's warm pool feature for production

### "File upload failed"

- Max file size: 20MB
- Supported formats: PDF, PNG, JPG, JPEG

## Cost Estimation

| Usage | GPU | Est. Monthly Cost |
|-------|-----|-------------------|
| 100 PDFs/month | L40S | ~$5-10 |
| 1000 PDFs/month | L40S | ~$50-100 |
| Production (24/7 warm) | L40S | ~$720 |

*Costs are estimates based on ~1 minute processing per PDF.*

## Links

- [Modal Documentation](https://modal.com/docs)
- [Modal Pricing](https://modal.com/pricing)
- [Dolphin GitHub](https://github.com/bytedance/Dolphin)
- [Dolphin Model](https://huggingface.co/ByteDance/Dolphin-v2)
