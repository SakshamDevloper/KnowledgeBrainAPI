"""KnowledgeBrain API — FastAPI Application Entry Point"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.utils.logging import setup_logging, get_logger

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown."""
    settings = get_settings()
    logger.info(f"Starting KnowledgeBrain API — environment: {settings.app_env}")

    # Initialize services
    try:
        from app.services.rag.embedder import QdrantService
        qdrant = QdrantService()
        await qdrant.ensure_collection()
        logger.info("Qdrant collection ready")
    except Exception as e:
        logger.warning(f"Qdrant initialization skipped: {e}")

    try:
        from app.services.graph.neo4j_service import Neo4jService
        neo4j = Neo4jService()
        await neo4j.connect()
        await neo4j.ensure_indexes()
        await neo4j.close()
        logger.info("Neo4j connection verified")
    except Exception as e:
        logger.warning(f"Neo4j initialization skipped: {e}")

    yield

    logger.info("Shutting down KnowledgeBrain API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="KnowledgeBrain API",
        description="AI for Industrial Knowledge Intelligence — Unified Asset & Operations Brain",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from app.routers import health, query, ingest, copilot, maintenance, compliance, lessons

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(copilot.router)
    app.include_router(maintenance.router)
    app.include_router(compliance.router)
    app.include_router(lessons.router)

    return app


# Configure logging on import
setup_logging()

# Create the application instance
app = create_app()
