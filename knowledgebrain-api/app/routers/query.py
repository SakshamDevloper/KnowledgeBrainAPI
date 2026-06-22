"""
KnowledgeBrain — Query Router
RAG query endpoint for natural language search.
"""

import time
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.query import QueryRequest, QueryResponse
from app.utils.logging import get_logger

logger = get_logger("router.query")
router = APIRouter(prefix="/query", tags=["Query"])

# Simple in-memory rate limiter
_rate_limits: dict[str, list[float]] = {}
RATE_LIMIT = 10  # max requests per minute per user


def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit (10 req/min)."""
    now = time.time()
    if user_id not in _rate_limits:
        _rate_limits[user_id] = []

    # Clean old entries
    _rate_limits[user_id] = [t for t in _rate_limits[user_id] if now - t < 60]

    if len(_rate_limits[user_id]) >= RATE_LIMIT:
        return False

    _rate_limits[user_id].append(now)
    return True


@router.post("", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the KnowledgeBrain knowledge base using natural language.

    Pipeline:
    1. Rate limit check
    2. Hybrid retrieval (semantic + keyword filter)
    3. Claude-powered answer synthesis
    4. Return answer with cited sources
    """
    settings = get_settings()

    # Rate limiting
    if not check_rate_limit(request.user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 10 requests per minute."
        )

    start_time = time.time()

    # Get services
    try:
        from app.services.rag.embedder import QdrantService
        qdrant_service = QdrantService()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector database unavailable: {e}")

    # Retrieve relevant chunks
    try:
        from app.services.rag.retriever import hybrid_retrieve
        chunks = await hybrid_retrieve(
            query=request.query,
            qdrant_service=qdrant_service,
            top_k=request.top_k,
            equipment_filter=request.equipment_filter or None,
            doc_type_filter=request.doc_type_filter or None,
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Synthesize answer
    try:
        anthropic_client = None
        if settings.anthropic_api_key:
            import anthropic
            anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        from app.services.rag.synthesizer import synthesize_answer
        response = await synthesize_answer(
            query=request.query,
            chunks=chunks,
            anthropic_client=anthropic_client,
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

    total_time = (time.time() - start_time) * 1000
    response.query_time_ms = round(total_time, 2)

    logger.info(
        f"Query answered in {total_time:.0f}ms: "
        f"'{request.query[:50]}...' -> {len(chunks)} chunks -> {response.confidence}"
    )

    return response
