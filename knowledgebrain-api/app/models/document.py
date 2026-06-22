"""
KnowledgeBrain — Document Models
Pydantic schemas for document ingestion and storage.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from uuid import uuid4


class DocType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"
    EMAIL = "email"
    MANUAL = "manual"
    PROCEDURE = "procedure"
    INCIDENT_REPORT = "incident_report"
    WORK_ORDER = "work_order"
    REGULATION = "regulation"
    UNKNOWN = "unknown"


class ExtractedEntity(BaseModel):
    """A single extracted entity with metadata."""
    entity_type: str = Field(..., description="Type: equipment_tag, process_param, regulatory_ref, person, date, document_id")
    value: str = Field(..., description="The extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence 0-1")
    source: str = Field(default="regex", description="Extraction method: regex or llm")
    context: str = Field(default="", description="Surrounding text for context")


class DocumentChunk(BaseModel):
    """A single chunk of a processed document."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    content: str
    page_num: Optional[int] = None
    chunk_index: int = 0
    source_file: str
    doc_type: DocType = DocType.UNKNOWN
    entities: list[ExtractedEntity] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentMetadata(BaseModel):
    """Metadata for an ingested document."""
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    doc_type: DocType = DocType.UNKNOWN
    file_size_bytes: int = 0
    page_count: int = 0
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    summary: str = ""


class IngestRequest(BaseModel):
    """Request body for document ingestion (when not using file upload)."""
    doc_type: DocType = DocType.UNKNOWN
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    doc_id: str
    filename: str
    doc_type: DocType
    chunk_count: int
    entities_found: dict[str, int] = Field(
        default_factory=dict,
        description="Count of entities by type: {equipment_tag: 5, regulatory_ref: 3, ...}"
    )
    summary: str
    processing_time_ms: float


class DocumentSearchResult(BaseModel):
    """A single document search result."""
    chunk_id: str
    doc_id: str
    content: str
    source_file: str
    page_num: Optional[int] = None
    doc_type: DocType
    relevance_score: float
    entities: list[ExtractedEntity] = Field(default_factory=list)
