"""
Loads the Chroma vector store that the notebook (Phase 2.3 / 2.7) persisted to disk,
and retrieves the most relevant chunks for a question.

The embedding model here MUST match the one used to build the store in the notebook,
or the vectors won't compare meaningfully. Both read from settings.EMBEDDING_MODEL
so there's a single source of truth.
"""
import logging

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_collection = None
_embedder: SentenceTransformer | None = None


def load_vector_store() -> None:
    """Called once at startup (see main.py lifespan) - loading per-request would be slow."""
    global _client, _collection, _embedder

    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
    _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    logger.info("Loading Chroma store from: %s", settings.VECTOR_STORE_PATH)
    _client = chromadb.PersistentClient(path=settings.VECTOR_STORE_PATH)
    _collection = _client.get_collection(settings.CHROMA_COLLECTION_NAME)
    logger.info(
        "Vector store loaded. %d chunks available.", _collection.count()
    )


def is_ready() -> bool:
    return _collection is not None and _embedder is not None


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """
    Returns a list of {"text": str, "source": str, "distance": float} dicts,
    the top_k most relevant chunks for the question.
    """
    if not is_ready():
        raise RuntimeError("Vector store not loaded - call load_vector_store() first")

    k = top_k or settings.TOP_K
    query_embedding = _embedder.encode([question]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "distance": dist,
        })
    return chunks
