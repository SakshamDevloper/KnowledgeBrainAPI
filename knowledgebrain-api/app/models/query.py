"""
KnowledgeBrain — Query Models
Pydantic schemas for RAG queries and responses.
"""

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for RAG query."""
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    user_id: str = Field(default="anonymous", description="User identifier for rate limiting")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to retrieve")
    equipment_filter: list[str] = Field(
        default_factory=list,
        description="Optional equipment tags to filter results"
    )
    doc_type_filter: list[str] = Field(
        default_factory=list,
        description="Optional document types to filter"
    )


class SourceReference(BaseModel):
    """A source citation in a query response."""
    filename: str
    page: Optional[int] = None
    excerpt: str = Field(default="", max_length=500)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    doc_type: str = ""


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    confidence: str = Field(default="MEDIUM", description="LOW | MEDIUM | HIGH")
    query_time_ms: float = 0.0
    chunks_retrieved: int = 0
