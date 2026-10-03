"""
Data Contracts and Validation Schemas for the NER Microservice.

This module defines Pydantic models for request parsing, input normalization,
payload constraint enforcement, and response serialization.

Key Architectural Roles:
1. Dynamic Input Polymorphism: Enables clients to submit either a single string
   or an array of strings under the same endpoint without schema mismatches.
2. Invariant Enforcement: Validates batch sizes and chunk lengths before payload
   data reaches the memory-heavy ML inference engine.
3. Serialized Schema Stability: Guarantees a consistent JSON contract regardless
   of whether GLiNER, spaCy, or another underlying model is active.
"""

from typing import List, Union, Optional
from pydantic import BaseModel, Field, field_validator


class Entity(BaseModel):
    """
    Represents an extracted named entity span within a text chunk.

    Attributes:
        text (str): The exact substring extracted from the text.
        label (str): Classification label (e.g., 'person', 'organization', 'location').
        start (int): 0-based character start index in the cleaned chunk (inclusive).
        end (int): 0-based character end index in the cleaned chunk (exclusive).
        score (Optional[float]): Model confidence score between 0.0 and 1.0.
            Populated by probabilistic models like GLiNER. May be None if
            unsupported by the underlying pipeline.
    """
    text: str
    label: str
    start: int
    end: int
    score: Optional[float] = None


class ChunkPrediction(BaseModel):
    """
    Container grouping all extracted entities for a single text chunk.

    Preserves the ordinal position of each chunk from the original request
    to simplify client-side correlation in batched workloads.

    Attributes:
        chunk_id (int): 0-indexed position corresponding to the chunk's order in the request.
        entities (List[Entity]): All detected entity spans exceeding the confidence threshold.
    """
    chunk_id: int
    entities: List[Entity]


class DynamicExtractionRequest(BaseModel):
    """
    Inbound API request payload for entity extraction.

    Supports both single-string queries (convenient for interactive testing)
    and arrays of chunks (optimized for high-throughput batch inference).

    Attributes:
        data (Union[str, List[str]]): The raw text or list of text segments to process.
        labels (Optional[List[str]]): Entity types to detect (e.g., ['person', 'organization']).
            Defaults to ['person', 'organization'] to match primary business use-cases.
        threshold (Optional[float]): Minimum probability score required to accept an entity.
            Defaults to 0.50 (balanced precision/recall baseline).
    """
    # Dynamically accepts either a single string OR an array of strings
    data: Union[str, List[str]] = Field(
        ..., 
        description="A single text string or an array of text chunks."
    )
    labels: Optional[List[str]] = Field(
        default=["person", "organization"],
        description="Entity labels to extract. Defaults to ['person', 'organization']."
    )
    threshold: Optional[float] = Field(
        default=0.5,
        description="Confidence threshold for predictions. Defaults to 0.5."
    )

    @field_validator('data')
    @classmethod
    def normalize_and_validate(cls, value):
        """
        Normalizes polymorphic input and validates performance boundaries.

        Business Rules & Safety Limits:
        1. Normalization: Wraps a single string into a single-item list so the
           downstream inference pipeline can always operate on an iterable batch.
        2. Non-empty: Prevents no-op invocations and resource allocation on empty payloads.
        3. Batch Limit (100 chunks): Protects against worker thread starvation and
           HTTP reverse-proxy timeouts (e.g., Nginx 60s gateway timeouts).
        4. Chunk Size Limit (2500 characters): Transformer self-attention complexity
           scales quadratically (O(n^2)) with sequence length. Chunks longer than 2500
           characters risk CUDA/CPU Out-Of-Memory (OOM) errors. The API rejects oversized
           chunks with an explicit error rather than silently truncating or auto-splitting,
           ensuring span offsets remain predictable for the caller.
        """
        # 1. Normalize: If it's a single string, wrap it in a list
        if isinstance(value, str):
            value = [value]
            
        # 2. Validate payload limits
        if not value:
            raise ValueError("Input data cannot be empty.")
        if len(value) > 100:
            raise ValueError("Batch limit exceeded: Maximum 100 chunks allowed per request.")
            
        # 3. Validate chunk size limit (no auto-splitting, just reject if oversized)
        for idx, chunk in enumerate(value):
            if len(chunk) > 2500:
                raise ValueError(f"Chunk at index {idx} exceeds the 2500 character limit.")
                
        return value


class ExtractionResponse(BaseModel):
    """
    Outbound API response structure returning chunk predictions.

    Attributes:
        results (List[ChunkPrediction]): Ordered predictions for each submitted chunk.
        total_processed (int): Total number of text chunks evaluated in this request.
    """
    results: List[ChunkPrediction]
    total_processed: int