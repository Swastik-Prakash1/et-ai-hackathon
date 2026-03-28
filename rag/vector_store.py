"""vector_store.py
=================================
ChromaDB vector store wrapper using PersistentClient(path="./chroma_db").
"""

import logging
import os
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION = "market_intelligence"

logger = logging.getLogger(__name__)


def _build_client_and_collection(collection_name: str = DEFAULT_COLLECTION):
    """Create a ChromaDB persistent client and return (client, collection).

    Args:
        collection_name: Collection name to open or create.

    Returns:
        Tuple of (PersistentClient instance, collection instance).
    """
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


_CLIENT, _COLLECTION = _build_client_and_collection(DEFAULT_COLLECTION)


def get_collection(collection_name: str = DEFAULT_COLLECTION):
    """Return an existing or new collection from the persistent client.

    Args:
        collection_name: Target collection.

    Returns:
        Chroma collection object.
    """
    try:
        if collection_name == DEFAULT_COLLECTION:
            return _COLLECTION
        return _CLIENT.get_or_create_collection(name=collection_name)
    except Exception as exc:
        logger.warning("Collection fetch failed, rebuilding client: %s", exc)
        client, collection = _build_client_and_collection(collection_name)
        return collection


def add_documents(
    documents: list[str],
    embeddings: list[list[float]],
    ids: list[str],
    metadatas: Optional[list[dict[str, Any]]] = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, Any]:
    """Add or update documents with embeddings in ChromaDB.

    Args:
        documents: Document text list.
        embeddings: Embeddings aligned to documents.
        ids: Unique IDs aligned to documents.
        metadatas: Optional metadata dict list.
        collection_name: Collection to upsert into.

    Returns:
        Status dictionary with count and collection.
    """
    if not (len(documents) == len(embeddings) == len(ids)):
        raise ValueError("documents, embeddings, and ids must be the same length")

    collection = get_collection(collection_name)

    if metadatas is None:
        metadatas = [{"ingested_at": datetime.now().isoformat(timespec="seconds")} for _ in documents]

    try:
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        return {
            "ok": True,
            "collection": collection_name,
            "count": len(ids),
        }
    except Exception as exc:
        logger.error("Failed to upsert documents into %s: %s", collection_name, exc)
        return {
            "ok": False,
            "collection": collection_name,
            "count": 0,
            "error": str(exc),
        }


def query_collection(
    query_embedding: list[float],
    n_results: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    where: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Query similar documents by vector embedding.

    Args:
        query_embedding: Query embedding vector.
        n_results: Number of nearest neighbors to return.
        collection_name: Collection to query.
        where: Optional metadata filter.

    Returns:
        Raw Chroma query result dictionary.
    """
    collection = get_collection(collection_name)

    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, int(n_results)),
            where=where,
        )
        return result
    except Exception as exc:
        logger.error("Vector query failed in %s: %s", collection_name, exc)
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "error": str(exc),
        }


def delete_documents(ids: list[str], collection_name: str = DEFAULT_COLLECTION) -> dict[str, Any]:
    """Delete documents by IDs from a collection.

    Args:
        ids: List of IDs to remove.
        collection_name: Target collection.

    Returns:
        Status dictionary.
    """
    collection = get_collection(collection_name)
    try:
        collection.delete(ids=ids)
        return {"ok": True, "deleted": len(ids), "collection": collection_name}
    except Exception as exc:
        logger.error("Delete failed in %s: %s", collection_name, exc)
        return {"ok": False, "deleted": 0, "collection": collection_name, "error": str(exc)}


def collection_count(collection_name: str = DEFAULT_COLLECTION) -> int:
    """Return number of vectors stored in a collection.

    Args:
        collection_name: Target collection.

    Returns:
        Integer item count.
    """
    collection = get_collection(collection_name)
    try:
        return int(collection.count())
    except Exception as exc:
        logger.error("Count failed for %s: %s", collection_name, exc)
        return 0


def smoke_test() -> dict[str, Any]:
    """Run vector store smoke test by upserting and querying sample records.

    Returns:
        Dictionary with count and top hit metadata.
    """
    from rag.embedder import embed_text

    docs = [
        "Promoter acquired 2L shares at Rs 450",
        "Quarterly earnings beat analyst estimate by 20.4 percent",
        "Management guided double-digit revenue growth for FY27",
    ]
    ids = [
        "smoke_001",
        "smoke_002",
        "smoke_003",
    ]
    embeddings = [embed_text(doc).tolist() for doc in docs]
    metadatas = [
        {"stock": "RELIANCE", "type": "insider_trade", "date": "2026-03-20"},
        {"stock": "RELIANCE", "type": "earnings", "date": "2026-03-20"},
        {"stock": "RELIANCE", "type": "commentary", "date": "2026-03-20"},
    ]

    upsert_status = add_documents(docs, embeddings, ids, metadatas=metadatas)
    query_vec = embed_text("insider buying in reliance").tolist()
    result = query_collection(query_vec, n_results=2)
    count = collection_count()

    return {
        "chroma_path": CHROMA_PATH,
        "upsert_ok": upsert_status.get("ok", False),
        "count": count,
        "top_ids": result.get("ids", [[]])[0],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(smoke_test())
