"""embedder.py
=================================
RAG embedding module using sentence-transformers/all-MiniLM-L6-v2.

Model is loaded once at module level and reused for all calls.
"""

import hashlib
import logging
import os
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

logger = logging.getLogger(__name__)


def _load_embedder_model() -> Any:
    """Load and return the sentence-transformers model once at import time."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL_NAME)
        return model
    except Exception as exc:
        logger.warning("Embedder model load failed, using hash fallback embeddings: %s", exc)
        return None


EMBEDDER = _load_embedder_model()


def _hash_embedding(text: str, dimension: int = EMBEDDING_DIMENSION) -> np.ndarray:
    """Create a deterministic fallback embedding when the model is unavailable.

    Args:
        text: Input text to embed.
        dimension: Output embedding dimension.

    Returns:
        Normalized deterministic vector of shape (dimension,).
    """
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dimension).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def embed_text(text: str) -> np.ndarray:
    """Embed one text string into a vector.

    Args:
        text: Text to convert into embedding.

    Returns:
        Numpy array embedding shape (384,).
    """
    if not text:
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

    if EMBEDDER is None:
        return _hash_embedding(text)

    try:
        vec = EMBEDDER.encode(text)
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr
    except Exception as exc:
        logger.warning("Embedding failed for text, switching to fallback hash embedding: %s", exc)
        return _hash_embedding(text)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed multiple texts in one call.

    Args:
        texts: List of input strings.

    Returns:
        Numpy array of shape (n, 384).
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)

    if EMBEDDER is None:
        vectors = [_hash_embedding(text) for text in texts]
        return np.vstack(vectors).astype(np.float32)

    try:
        vectors = EMBEDDER.encode(texts)
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception as exc:
        logger.warning("Batch embedding failed, falling back to per-text hash embeddings: %s", exc)
        vectors = [_hash_embedding(text) for text in texts]
        return np.vstack(vectors).astype(np.float32)


def smoke_test() -> dict:
    """Run a quick embedder smoke test.

    Returns:
        Dictionary with shape and model metadata.
    """
    sample = "Promoter bought 2L shares at Rs 450"
    vec = embed_text(sample)
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "embedder_loaded": EMBEDDER is not None,
        "dimension": int(vec.shape[0]),
        "norm": float(np.linalg.norm(vec)),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(smoke_test())
