import logging

from fastapi import APIRouter, HTTPException

from app.schemas.query import QueryRequest, QueryResponse
from app.services import generation, retrieval

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "vector_store_ready": retrieval.is_ready()}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not retrieval.is_ready():
        raise HTTPException(status_code=503, detail="Vector store is not loaded yet")

    try:
        chunks = retrieval.retrieve(request.question)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if not chunks:
        return QueryResponse(
            answer="I couldn't find anything relevant in the course material for that question.",
            sources=[],
        )

    try:
        answer = generation.generate_answer(request.question, chunks)
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach the local Ollama model: {exc}",
        ) from exc

    sources = sorted({chunk["source"] for chunk in chunks})
    return QueryResponse(answer=answer, sources=sources)
