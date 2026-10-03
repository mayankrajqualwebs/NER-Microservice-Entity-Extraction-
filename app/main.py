"""
Dynamic Named Entity Recognition (NER) Microservice Entry Point.

This module exposes a RESTful FastAPI service for high-performance named entity
extraction from unstructured text documents. It functions as both the HTTP API
controller and the model lifecycle manager.

Architecture & Data Flow:
    1. Inbound Request:
       HTTP Client -> POST /extract with JSON payload (DynamicExtractionRequest).
    2. Request Validation & Normalization:
       Pydantic validates payload constraints (max 100 chunks, max 2500 chars/chunk)
       and normalizes input into a List[str].
    3. Preprocessing:
       Each chunk is sanitized via `clean_text()` to normalize spacing and strip noise.
    4. Inference Execution:
       `ModelRunner.predict()` invokes the active model engine (GLiNER or spaCy)
       using specified target entity labels (e.g. ['person', 'organization']).
    5. Response Serialization:
       Entities with start/end offsets and confidence scores are serialized into
       an `ExtractionResponse` JSON contract.

Model Hot-Swapping Contract:
    The microservice is engineered to support seamless model transitions. When a
    new custom model finishes training, its artifacts (e.g. weights, configs,
    tokenizers) can be placed directly into the `model/` directory. Upon startup
    or restart, `ModelRunner` detects the artifact format and switches engines
    automatically without requiring code changes or schema migrations.

Configuration & Environment Variables:
    MODEL_PATH (str): Path to local model directory (defaults to "model").
    FALLBACK_MODEL (str): Remote Hugging Face model repository ID used when local
                          weights are absent ("urchade/gliner_medium-v2.1").
"""

import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from app.schemas import DynamicExtractionRequest, ExtractionResponse, ChunkPrediction, Entity
from app.utils import clean_text

# Configure service logging format
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ner_microservice")

# Model configuration paths
MODEL_DIR = os.getenv("MODEL_PATH", "model")
FALLBACK_MODEL = "urchade/gliner_medium-v2.1"


class ModelRunner:
    """
    Universal Model Container and Inference Orchestrator.

    Encapsulates model discovery, lazy/startup loading, engine abstraction
    (GLiNER transformer vs. spaCy pipeline), and unified entity prediction.

    Attributes:
        model_path (str): Filesystem path to the model directory.
        model (Any): The instantiated machine learning model instance.
        model_type (str): Active engine identifier ('gliner' or 'spacy').
        model_source (str): Human-readable provenance of the active model
            (e.g., 'local:model', 'urchade/gliner_medium-v2.1', 'spacy:en_core_web_sm').
    """

    def __init__(self, model_path: str = MODEL_DIR):
        """
        Initializes the model container and loads the model into memory.

        Args:
            model_path (str): Filesystem directory where local model artifacts reside.
        """
        self.model_path = model_path
        self.model = None
        self.model_type = None  # "gliner" or "spacy"
        self.model_source = None
        self.load()

    def load(self):
        """
        Discovers, detects, and initializes the appropriate NER engine.

        Detection & Fallback Strategy:
            Tier 1 (Local GLiNER): Checks if `model_path` contains GLiNER configuration
                   or weight files (`gliner_config.json`, `model.safetensors`, `pytorch_model.bin`).
                   If found, loads via `GLiNER.from_pretrained(model_path)`.
            Tier 2 (Local spaCy): Checks if `model_path` contains a spaCy configuration
                   or metadata (`config.cfg`, `meta.json`). If found, loads via `spacy.load(model_path)`.
            Tier 3 (Hugging Face GLiNER Fallback): If `model_path` is empty or missing,
                   downloads/loads the base pre-trained model `urchade/gliner_medium-v2.1`.
            Tier 4 (spaCy Fallback): If GLiNER fails to load (e.g., missing dependencies or network block),
                   falls back to spaCy's `en_core_web_sm` pipeline as an emergency safety net.

        Side Effects:
            Loads heavy model weights into host/GPU memory (typically 700MB - 1.5GB RAM).
        """
        # 1. Check if model_path contains a GLiNER model
        gliner_indicators = ["gliner_config.json", "model.safetensors", "pytorch_model.bin"]
        has_gliner_files = (
            os.path.isdir(self.model_path) and
            any(os.path.exists(os.path.join(self.model_path, f)) for f in gliner_indicators)
        )

        # 2. Check if model_path contains a spaCy model
        spacy_indicators = ["config.cfg", "meta.json"]
        has_spacy_files = (
            os.path.isdir(self.model_path) and
            any(os.path.exists(os.path.join(self.model_path, f)) for f in spacy_indicators)
        )

        # Primary route: Local fine-tuned or downloaded GLiNER model
        if has_gliner_files:
            try:
                from gliner import GLiNER
                logger.info(f"Loading local GLiNER model from '{self.model_path}'...")
                self.model = GLiNER.from_pretrained(self.model_path)
                self.model_type = "gliner"
                self.model_source = f"local:{self.model_path}"
                logger.info(f"Loaded local GLiNER model from '{self.model_path}'.")
                return
            except Exception as e:
                logger.warning(f"Could not load local GLiNER model from '{self.model_path}': {e}")

        # Secondary route: Local spaCy model pipeline
        elif has_spacy_files:
            try:
                import spacy
                logger.info(f"Loading local spaCy model from '{self.model_path}'...")
                self.model = spacy.load(self.model_path)
                self.model_type = "spacy"
                self.model_source = f"local_spacy:{self.model_path}"
                logger.info(f"Loaded local spaCy model from '{self.model_path}'.")
                return
            except Exception as e:
                logger.warning(f"Could not load local spaCy model from '{self.model_path}': {e}")

        # Fallback route: Remote Hugging Face base model
        try:
            from gliner import GLiNER
            logger.info(f"Loading fallback GLiNER model '{FALLBACK_MODEL}'...")
            self.model = GLiNER.from_pretrained(FALLBACK_MODEL)
            self.model_type = "gliner"
            self.model_source = FALLBACK_MODEL
            logger.info(f"Fallback GLiNER model '{FALLBACK_MODEL}' loaded.")
        except Exception as e:
            logger.error(f"Failed to load fallback GLiNER model: {e}")
            # Final emergency fallback: spaCy lightweight English model
            import spacy
            try:
                self.model = spacy.load("en_core_web_sm")
            except OSError:
                import spacy.cli
                spacy.cli.download("en_core_web_sm")
                self.model = spacy.load("en_core_web_sm")
            self.model_type = "spacy"
            self.model_source = "spacy:en_core_web_sm"

    def predict(
        self,
        chunks: List[str],
        labels: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> List[ChunkPrediction]:
        """
        Executes entity extraction across a list of text chunks.

        Normalizes differences in prediction signatures between GLiNER
        (zero-shot span classification) and spaCy (pipeline token tagger),
        outputting a unified list of ChunkPrediction objects.

        Args:
            chunks (List[str]): Cleaned text chunks to evaluate.
            labels (Optional[List[str]]): Target entity types. Defaults to
                ['person', 'organization'] if None or empty.
            threshold (float): Minimum confidence score cutoff for entities.
                Defaults to 0.5.

        Returns:
            List[ChunkPrediction]: Ordered predictions with entity boundaries and confidence.
        """
        if not labels:
            labels = ["person", "organization"]

        results = []
        if self.model_type == "gliner":
            for idx, chunk in enumerate(chunks):
                # Optimization: Skip empty/whitespace-only chunks to conserve compute
                if not chunk.strip():
                    results.append(ChunkPrediction(chunk_id=idx, entities=[]))
                    continue

                # GLiNER zero-shot entity extraction
                raw_entities = self.model.predict_entities(
                    chunk,
                    labels=labels,
                    threshold=threshold
                )
                # Map raw GLiNER dictionaries to validated Entity models
                entities = [
                    Entity(
                        text=ent["text"].strip(),
                        label=ent["label"],
                        start=ent["start"],
                        end=ent["end"],
                        # Round score to 4 decimal places for clean JSON output
                        score=round(float(ent["score"]), 4) if "score" in ent else None
                    )
                    for ent in raw_entities
                ]
                results.append(ChunkPrediction(chunk_id=idx, entities=entities))

        elif self.model_type == "spacy":
            # Process chunks in parallel using spaCy pipe
            docs = list(self.model.pipe(chunks, batch_size=32))
            labels_set = {lbl.lower() for lbl in labels}
            for idx, doc in enumerate(docs):
                entities = [
                    Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        score=1.0  # spaCy default pipelines do not yield per-entity probability scores
                    )
                    for ent in doc.ents
                    if ent.label_.lower() in labels_set
                ]
                results.append(ChunkPrediction(chunk_id=idx, entities=entities))

        return results


# Global model runner instance loaded once at service startup
runner = ModelRunner()

# Initialize FastAPI application
app = FastAPI(
    title="Dynamic NER Microservice",
    description="Production-ready microservice for Named Entity Recognition using GLiNER / Custom Models",
    version="1.0.0"
)


@app.get("/health")
async def health():
    """
    Service health and model readiness probe.

    Used by Kubernetes liveness/readiness probes, Docker healthchecks,
    and monitoring dashboards to inspect service availability and verify
    which model backend is actively serving traffic.

    Returns:
        dict: Health status ('healthy'), active model type, and model source path.
    """
    return {
        "status": "healthy",
        "model_type": runner.model_type,
        "model_source": runner.model_source
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_entities(payload: DynamicExtractionRequest):
    """
    Extracts named entities from submitted text chunk(s).

    Workflow:
        1. Accesses validated and normalized payload data (List[str]).
        2. Applies whitespace and noise sanitization via clean_text.
        3. Executes entity prediction through the active ModelRunner.
        4. Packages predictions with chunk IDs and accounting totals.

    Args:
        payload (DynamicExtractionRequest): Request containing text data,
            optional custom entity labels, and confidence threshold.

    Returns:
        ExtractionResponse: Structured results containing chunk predictions
            and total processed chunk count.
    """
    # payload.data is guaranteed to be a List[str] due to Pydantic field validation
    cleaned_chunks = [clean_text(chunk) for chunk in payload.data]
    
    results = runner.predict(
        chunks=cleaned_chunks,
        labels=payload.labels,
        threshold=payload.threshold
    )
    
    return ExtractionResponse(
        results=results,
        total_processed=len(cleaned_chunks)
    )