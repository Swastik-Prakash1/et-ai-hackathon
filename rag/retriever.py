"""retriever.py
=================================
RAG retrieval pipeline combining rag.embedder and rag.vector_store.
"""

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_TOP_K = 5
DEFAULT_COLLECTION = "market_intelligence"

logger = logging.getLogger(__name__)


def _distance_to_relevance(distance: Optional[float]) -> float:
    """Convert Chroma distance to a bounded relevance score in [0, 1].

    Args:
        distance: Raw vector distance value from Chroma.

    Returns:
        Relevance score, where higher is better.
    """
    if distance is None:
        return 0.0
    dist = float(distance)
    if dist < 0:
        dist = 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + dist)))


def retrieve_context(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = DEFAULT_COLLECTION,
    where: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Retrieve top-k context chunks for a natural language query.

    Args:
        query: User query string.
        top_k: Number of chunks to return.
        collection_name: Chroma collection name.
        where: Optional metadata filter.

    Returns:
        Retrieval payload containing query and ranked context chunks.
    """
    from rag.embedder import embed_text
    from rag.vector_store import query_collection

    clean_query = query.strip()
    if not clean_query:
        return {
            "query": query,
            "collection": collection_name,
            "top_k": int(top_k),
            "results": [],
        }

    query_vec = embed_text(clean_query).tolist()
    raw = query_collection(
        query_embedding=query_vec,
        n_results=max(1, int(top_k)),
        collection_name=collection_name,
        where=where,
    )

    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    results: list[dict[str, Any]] = []
    for idx in range(min(len(ids), len(docs), len(metas), len(dists))):
        distance = dists[idx]
        relevance = _distance_to_relevance(distance)
        item = {
            "id": ids[idx],
            "text": docs[idx],
            "metadata": metas[idx] or {},
            "distance": float(distance) if distance is not None else None,
            "relevance": round(relevance, 4),
            "source": (metas[idx] or {}).get("source", "unknown"),
        }
        results.append(item)

    results.sort(key=lambda x: x["relevance"], reverse=True)

    return {
        "query": clean_query,
        "collection": collection_name,
        "top_k": int(top_k),
        "results": results,
    }


def build_reasoning_context(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = DEFAULT_COLLECTION,
    where: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Build compact context list for reasoning_agent input.

    Args:
        query: User or system query.
        top_k: Number of chunks.
        collection_name: Target collection.
        where: Optional metadata filter.

    Returns:
        List of dictionaries with text and source keys.
    """
    payload = retrieve_context(query, top_k=top_k, collection_name=collection_name, where=where)
    context: list[dict[str, Any]] = []
    for item in payload.get("results", []):
        metadata = item.get("metadata", {})
        source = metadata.get("source") or metadata.get("type") or "market_intelligence"
        context.append(
            {
                "text": item.get("text", ""),
                "source": str(source),
                "relevance": item.get("relevance", 0.0),
                "metadata": metadata,
            }
        )
    return context


def smoke_test() -> dict[str, Any]:
    """Run retrieval smoke test after seeding a few records in vector store.

    Returns:
        Dictionary with retrieval summary.
    """
    from rag.embedder import embed_text
    from rag.vector_store import add_documents

    docs = [
        "Promoter net buying in RELIANCE increased this week.",
        "RELIANCE reported quarterly earnings beat versus analyst estimate.",
        "Management tone shifted positive with stronger growth guidance.",
    ]
    ids = ["retrieval_001", "retrieval_002", "retrieval_003"]
    embeds = [embed_text(doc).tolist() for doc in docs]
    metas = [
        {"stock": "RELIANCE", "type": "insider_trade", "source": "SEBI PIT"},
        {"stock": "RELIANCE", "type": "earnings", "source": "yfinance"},
        {"stock": "RELIANCE", "type": "commentary", "source": "earnings_call"},
    ]

    add_documents(docs, embeds, ids, metas, collection_name=DEFAULT_COLLECTION)
    payload = retrieve_context("show me insider buying and earnings beat", top_k=3)

    return {
        "query": payload.get("query"),
        "result_count": len(payload.get("results", [])),
        "top_result": payload.get("results", [None])[0],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(smoke_test())
