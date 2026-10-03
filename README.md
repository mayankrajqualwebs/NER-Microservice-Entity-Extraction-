# Dynamic NER Microservice (Entity Extraction)

A production-ready REST microservice for high-accuracy **Named Entity Recognition (NER)** built with **FastAPI**, **GLiNER**, and **Docker**.

---

## Architecture Overview

```
                          ┌───────────────────────────────────────────────┐
                          │               Incoming Request                │
                          │             POST /extract (JSON)              │
                          └──────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                          ┌───────────────────────────────────────────────┐
                          │         Pydantic Request Validation           │
                          │  - Single string or List[str] polymorphism   │
                          │  - Max 100 chunks limit                       │
                          │  - Max 2500 chars/chunk safety limit          │
                          └──────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                          ┌───────────────────────────────────────────────┐
                          │              Text Preprocessing               │
                          │  - clean_text(): Whitespace normalization    │
                          └──────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                          ┌───────────────────────────────────────────────┐
                          │           ModelRunner Engine Layer            │
                          │  Tier 1: Local GLiNER (model/ directory)      │
                          │  Tier 2: Local spaCy (model/ directory)       │
                          │  Tier 3: Remote Hugging Face fallback         │
                          │  Tier 4: Local spaCy emergency fallback       │
                          └──────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                          ┌───────────────────────────────────────────────┐
                          │           Structured API Response             │
                          │  - ChunkPredictions with start/end offsets    │
                          │  - Classification labels and confidence score │
                          │  - Total accounting metadata                  │
                          └───────────────────────────────────────────────┘
```

---

## Key Features

- **Dynamic Zero-Shot Extraction**: Powered by GLiNER (`urchade/gliner_medium-v2.1`), allowing arbitrary entity types (`person`, `organization`, `location`, `date`, etc.) to be extracted on the fly without retraining.
- **Zero-Code-Change Model Swapping**: Place fine-tuned weights into `model/`, and `ModelRunner` detects and loads the new model automatically upon restart.
- **Resilient Fallback Pipeline**: Automatically falls back to remote Hugging Face weights or local spaCy if local model files are absent or corrupted.
- **High-Throughput Batch Processing**: Accepts individual strings or batches up to 100 chunks per request with configurable confidence thresholds.
- **Memory Safety Guardrails**: Hard limit of 2,500 characters per chunk prevents transformer quadratic attention ($O(n^2)$) memory explosion and Out-Of-Memory (OOM) errors.
- **Containerized for Production**: Optimized multi-stage Docker build ready for Kubernetes, AWS ECS, or standalone Docker engines.

---

## Directory Structure

```
.
├── Dockerfile                  # Production container definition
├── .dockerignore               # Build context exclusions
├── .gitignore                  # Git tracking rules (excludes heavy model weights)
├── .env.example                # Environment configuration template
├── README.md                   # Project documentation
├── requirements.txt            # Python package dependencies with explanations
├── app/
│   ├── __init__.py             # Application package marker & version
│   ├── main.py                 # FastAPI application, routes, and ModelRunner
│   ├── schemas.py              # Pydantic data contracts and validation logic
│   └── utils.py                # Text normalization and preprocessing utilities
└── model/
    ├── README.md               # Model management and export instructions
    ├── gliner_config.json      # Model architecture configuration
    ├── tokenizer.json          # Fast tokenizer vocabulary
    ├── tokenizer_config.json   # Tokenizer rules
    ├── model.safetensors       # Model weights (ignored in Git; see Model Guide)
    └── pytorch_model.bin       # PyTorch binary weights (ignored in Git)
```

---

## Quickstart (Local Development)

### 1. Prerequisites
- Python 3.10+ installed
- Virtual environment tool (`venv`)

### 2. Environment Setup
```bash
# Clone the repository
git clone <your-github-repo-url>
cd "Dockerize Microservice - Entity Extraction"

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The interactive Swagger API documentation is available at: **`http://localhost:8000/docs`**

---

## Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t ner-microservice:latest .
```

### 2. Run the Container
```bash
docker run -d -p 8000:8000 --name ner-service ner-microservice:latest
```

### 3. Mounting Custom Models at Runtime
To swap models dynamically without rebuilding the image, mount an external model directory via Docker volume:
```bash
docker run -d -p 8000:8000 \
  -v /path/to/my_trained_model:/app/model \
  --name ner-service ner-microservice:latest
```

---

## API Reference

### 1. Health & Readiness Probe
`GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_type": "gliner",
  "model_source": "local:model"
}
```

---

### 2. Entity Extraction
`POST /extract`

#### Request Parameters
| Field | Type | Required | Default | Description |
| :--- | :--- | :---: | :---: | :--- |
| `data` | `string` \| `array[string]` | **Yes** | — | Single text string or list of text chunks (max 100 chunks, max 2500 chars/chunk). |
| `labels` | `array[string]` | No | `["person", "organization"]` | Entity categories to extract. |
| `threshold` | `float` | No | `0.5` | Minimum confidence score threshold (0.0 to 1.0). |

#### Example Request: Single Chunk
```bash
curl -X POST "http://localhost:8000/extract" \
     -H "Content-Type: application/json" \
     -d '{
       "data": "Dr. George D. Green is the CEO of UNION DENTAL CORP."
     }'
```

#### Example Request: Batch with Custom Labels
```bash
curl -X POST "http://localhost:8000/extract" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         "Dr. George D. Green attended the meeting in Florida.",
         "Apple Inc. was founded by Steve Jobs."
       ],
       "labels": ["person", "organization", "location"],
       "threshold": 0.45
     }'
```

#### Example Response
```json
{
  "results": [
    {
      "chunk_id": 0,
      "entities": [
        {
          "text": "Dr. George D. Green",
          "label": "person",
          "start": 0,
          "end": 19,
          "score": 0.972
        },
        {
          "text": "Florida",
          "label": "location",
          "start": 45,
          "end": 52,
          "score": 0.968
        }
      ]
    },
    {
      "chunk_id": 1,
      "entities": [
        {
          "text": "Apple Inc.",
          "label": "organization",
          "start": 0,
          "end": 10,
          "score": 0.985
        },
        {
          "text": "Steve Jobs",
          "label": "person",
          "start": 26,
          "end": 36,
          "score": 0.992
        }
      ]
    }
  ],
  "total_processed": 2
}
```

---

## Model Hot-Swapping Guide

When a newly trained custom model is ready:

1. **Save Model Artifacts**:
   Ensure your training pipeline exports model artifacts using Hugging Face / GLiNER standard formatting:
   ```python
   model.save_pretrained("model/")
   ```
2. **Replace Folder Contents**:
   Replace the files in `model/` with your newly trained model (`gliner_config.json`, `model.safetensors` or `pytorch_model.bin`, `tokenizer.json`).
3. **Restart Service**:
   Restart the service or container. `ModelRunner` detects the new artifacts and loads them automatically.
4. **No Code Modification Needed**:
   All endpoints, validation schemas, and downstream pipelines remain intact.

---

## GitHub & Large File Policy

- **File Size Limit**: GitHub enforces a strict 100 MB limit per file. Heavy weights (`*.safetensors`, `*.bin`, `*.pt`) exceed this threshold and are ignored via [`.gitignore`](file:///.gitignore).
- **Git LFS Option**: If you prefer tracking model weights in Git, install Git LFS:
  ```bash
  git lfs install
  git lfs track "model/*.safetensors"
  git lfs track "model/*.bin"
  git add .gitattributes
  ```
- **Recommended Production Practice**: Keep configuration and tokenizers in the repository while pulling model weights from cloud storage (AWS S3, GCP Cloud Storage) or mounting host volumes during container deployment.
