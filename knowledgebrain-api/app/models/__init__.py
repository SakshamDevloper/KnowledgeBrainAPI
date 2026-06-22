"""KnowledgeBrain — Models Package"""

from app.models.document import (
    DocType,
    ExtractedEntity,
    DocumentChunk,
    DocumentMetadata,
    IngestRequest,
    IngestResponse,
    DocumentSearchResult,
)
from app.models.query import QueryRequest, QueryResponse, SourceReference
from app.models.copilot import CopilotRequest, CopilotResponse, CopilotSource

__all__ = [
    "DocType",
    "ExtractedEntity",
    "DocumentChunk",
    "DocumentMetadata",
    "IngestRequest",
    "IngestResponse",
    "DocumentSearchResult",
    "QueryRequest",
    "QueryResponse",
    "SourceReference",
    "CopilotRequest",
    "CopilotResponse",
    "CopilotSource",
]
