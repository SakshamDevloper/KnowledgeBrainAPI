"""
KnowledgeBrain — Ingest Router
Document ingestion endpoints for PDFs, spreadsheets, and emails.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from app.config import get_settings
from app.models.document import DocType, IngestResponse
from app.utils.logging import get_logger

logger = get_logger("router.ingest")
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


def get_anthropic_client():
    """Get Anthropic client if API key is configured."""
    settings = get_settings()
    if settings.anthropic_api_key:
        import anthropic
        return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return None


def get_qdrant_service():
    """Get Qdrant service instance."""
    try:
        from app.services.rag.embedder import QdrantService
        return QdrantService()
    except Exception as e:
        logger.warning(f"Qdrant not available: {e}")
        return None


def get_graph_builder():
    """Get graph builder instance."""
    try:
        from app.services.graph.neo4j_service import Neo4jService
        from app.services.graph.graph_builder import GraphBuilder
        neo4j = Neo4jService()
        # Note: connect() is async, we'll handle in the endpoint
        return neo4j, GraphBuilder(neo4j)
    except Exception as e:
        logger.warning(f"Neo4j not available: {e}")
        return None, None


def detect_doc_type(filename: str, explicit_type: Optional[str] = None) -> DocType:
    """Auto-detect document type from filename extension."""
    if explicit_type:
        try:
            return DocType(explicit_type)
        except ValueError:
            pass

    lower = filename.lower()
    if lower.endswith(".pdf"):
        return DocType.PDF
    elif lower.endswith((".xlsx", ".xls", ".csv")):
        return DocType.SPREADSHEET
    elif lower.endswith((".eml", ".msg")):
        return DocType.EMAIL
    elif lower.endswith(".txt"):
        return DocType.EMAIL  # Assume text files are email exports
    return DocType.UNKNOWN


@router.post("", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(None),
):
    """
    Ingest a document (PDF, Excel/CSV, or email) into KnowledgeBrain.

    The pipeline:
    1. Detect document type
    2. Extract text and metadata
    3. Chunk and extract entities
    4. Generate AI summary
    5. Store embeddings in Qdrant
    6. Build knowledge graph nodes in Neo4j
    """
    settings = get_settings()

    # Validate file size
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB"
        )

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    filename = file.filename or "unknown"
    detected_type = detect_doc_type(filename, doc_type)

    logger.info(f"Ingesting document: {filename} (type: {detected_type.value})")

    # Get services
    anthropic_client = get_anthropic_client()
    qdrant_service = get_qdrant_service()

    # Initialize Qdrant collection
    if qdrant_service:
        try:
            await qdrant_service.ensure_collection()
        except Exception as e:
            logger.warning(f"Qdrant collection setup failed: {e}")
            qdrant_service = None

    # Initialize Neo4j
    neo4j_service, graph_builder = get_graph_builder()
    if neo4j_service:
        try:
            await neo4j_service.connect()
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")
            graph_builder = None

    try:
        if detected_type == DocType.PDF:
            from app.services.ingestion.pdf_processor import process_pdf
            result = await process_pdf(
                file_bytes=file_bytes,
                filename=filename,
                doc_type=detected_type,
                anthropic_client=anthropic_client,
                qdrant_service=qdrant_service,
                graph_builder=graph_builder,
            )

        elif detected_type == DocType.SPREADSHEET:
            from app.services.ingestion.spreadsheet_processor import process_spreadsheet
            result = await process_spreadsheet(
                file_bytes=file_bytes,
                filename=filename,
                doc_type=detected_type,
                anthropic_client=anthropic_client,
                qdrant_service=qdrant_service,
                graph_builder=graph_builder,
            )

        elif detected_type == DocType.EMAIL:
            from app.services.ingestion.email_processor import process_email
            result = await process_email(
                file_bytes=file_bytes,
                filename=filename,
                doc_type=detected_type,
                anthropic_client=anthropic_client,
                qdrant_service=qdrant_service,
                graph_builder=graph_builder,
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported document type: {detected_type.value}. Supported: pdf, spreadsheet (xlsx/csv), email (eml/txt)"
            )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if neo4j_service:
            await neo4j_service.close()
