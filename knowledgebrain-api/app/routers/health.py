"""
KnowledgeBrain — Health Router
System health checks and status.
"""

from fastapi import APIRouter, Depends
from app.config import get_settings, Settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "service": "KnowledgeBrain API",
        "version": "1.0.0",
    }


@router.get("/detailed")
async def detailed_health():
    """Detailed health check including dependencies."""
    settings = get_settings()
    status = {
        "status": "healthy",
        "service": "KnowledgeBrain API",
        "version": "1.0.0",
        "environment": settings.app_env,
        "dependencies": {},
    }

    # Check Qdrant
    try:
        from app.services.rag.embedder import QdrantService
        qdrant = QdrantService()
        stats = await qdrant.get_collection_stats()
        status["dependencies"]["qdrant"] = {
            "status": "connected",
            "points_count": stats.get("points_count", 0),
        }
    except Exception as e:
        status["dependencies"]["qdrant"] = {"status": "error", "error": str(e)}

    # Check Neo4j
    try:
        from app.services.graph.neo4j_service import Neo4jService
        neo4j = Neo4jService()
        await neo4j.connect()
        graph_stats = await neo4j.get_graph_stats()
        await neo4j.close()
        status["dependencies"]["neo4j"] = {
            "status": "connected",
            "node_counts": graph_stats,
        }
    except Exception as e:
        status["dependencies"]["neo4j"] = {"status": "error", "error": str(e)}

    # Check Anthropic
    status["dependencies"]["anthropic"] = {
        "status": "configured" if settings.anthropic_api_key else "not_configured",
        "model": settings.anthropic_model,
    }

    return status
