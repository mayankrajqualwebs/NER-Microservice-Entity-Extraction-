"""
Text Preprocessing Utilities for NER Inference.

This module provides text cleaning and normalization helpers executed prior
to passing input strings to the machine learning inference engine. Transformer-based
tokenizers (such as DeBERTa-v3 in GLiNER) are sensitive to irregular spacing,
unprintable control characters, and redundant whitespace, which can distort subword
tokenization and span boundary offsets.
"""

def clean_text(text: str) -> str:
    """
    Normalizes whitespace and removes unwanted hidden characters from a text chunk.

    Why this is needed:
    Raw document chunks extracted from PDFs, OCR, or web scraping often contain
    erratic whitespace (e.g., repeated spaces, non-standard line breaks, tabs).
    This function standardizes all whitespace sequences into single ASCII spaces
    and strips leading/trailing margins.

    Note on Character Offsets:
    Collapsing whitespace modifies character indexes. If downstream systems rely
    on strict byte-level alignment with raw source documents, client applications
    should send pre-segmented, pre-cleaned chunks.

    Args:
        text (str): The raw text chunk to sanitize.

    Returns:
        str: A normalized string with single-space separators, or an empty string
             if input is falsy.
    """
    if not text:
        return ""
    # Replaces tabs, newlines, and multiple spaces with a single space
    return " ".join(text.split())