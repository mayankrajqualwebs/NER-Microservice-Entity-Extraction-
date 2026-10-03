# Model Artifacts Directory

This directory houses the machine learning model weights, configurations, and tokenizer files utilized by the NER microservice.

## Current Default Architecture
- **Base Model**: `urchade/gliner_medium-v2.1` (bidirectional transformer based on DeBERTa-v3)
- **Engine**: GLiNER (Generalist and Lightweight Named Entity Recognition)

## Artifact Files
When populated locally, this folder contains:
- `gliner_config.json`: Model architecture, hyperparameters, and class mapping.
- `model.safetensors` / `pytorch_model.bin`: Model weights (ignored in Git due to file size >100 MB).
- `tokenizer.json` & `tokenizer_config.json`: Fast subword tokenizer vocabulary and rules.

## Replacing with a Custom-Trained Model
When your custom model training completes:
1. Export your fine-tuned model via:
   ```python
   model.save_pretrained("model/")
   ```
2. Place the resulting configuration, tokenizer, and weight files into this `model/` folder.
3. Restart or redeploy the microservice.

The `ModelRunner` in `app/main.py` will automatically detect the updated model and load it with **zero code changes**.
