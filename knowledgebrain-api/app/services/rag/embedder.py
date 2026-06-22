"""
KnowledgeBrain — Embedder Service
Manages the sentence-transformer model and Qdrant vector storage.
"""

import asyncio
from typing import Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import get_settings
from app.models.document import DocumentChunk
from app.utils.logging import get_logger

logger = get_logger("embedder")

# Singleton for the embedding model (lazy loaded)
_embedding_model = None
_model_lock = asyncio.Lock()


async def get_embedding_model():
    """Lazy-load the sentence transformer model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        async with _model_lock:
            if _embedding_model is None:
                settings = get_settings()
                logger.info(f"Loading embedding model: {settings.embedding_model}")
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(settings.embedding_model)
                logger.info("Embedding model loaded successfully")
    return _embedding_model


async def embed_text(text: str) -> list[float]:
    """Embed a single text string."""
    model = await get_embedding_model()
    # Run in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, lambda: model.encode(text).tolist())
    return embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts."""
    model = await get_embedding_model()
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None, lambda: model.encode(texts, show_progress_bar=False).tolist()
    )
    return embeddings


class QdrantService:
    """Manages Qdrant vector database operations."""

    def __init__(self, client: Optional[QdrantClient] = None):
        settings = get_settings()
        if client:
            self.client = client
        else:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
        self.collection_name = settings.qdrant_collection_name
        self.dimension = settings.embedding_dimension

    async def ensure_collection(self):
        """Create the collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            if self.collection_name not in names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.dimension,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")
            raise

    async def store_chunks(self, chunks: list[DocumentChunk]):
        """Embed and store document chunks in Qdrant."""
        if not chunks:
            return

        # Embed all chunk contents
        texts = [chunk.content for chunk in chunks]
        embeddings = await embed_texts(texts)

        # Build Qdrant points
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "source_file": chunk.source_file,
                "page_num": chunk.page_num,
                "chunk_index": chunk.chunk_index,
                "doc_type": chunk.doc_type.value,
                "equipment_tags": chunk.metadata.get("equipment_tags", []),
                "regulatory_refs": chunk.metadata.get("regulatory_refs", []),
                "created_at": chunk.created_at.isoformat(),
            }

            points.append(qdrant_models.PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload=payload,
            ))

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Stored {len(points)} chunks in Qdrant")

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        equipment_filter: Optional[list[str]] = None,
        doc_type_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Search Qdrant for similar chunks.
        Optionally filter by equipment tags or document types.
        """
        # Build filter conditions
        must_conditions = []

        if equipment_filter:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="equipment_tags",
                    match=qdrant_models.MatchAny(any=equipment_filter),
                )
            )

        if doc_type_filter:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="doc_type",
                    match=qdrant_models.MatchAny(any=doc_type_filter),
                )
            )

        query_filter = None
        if must_conditions:
            query_filter = qdrant_models.Filter(must=must_conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            {
                "chunk_id": hit.payload.get("chunk_id", hit.id),
                "doc_id": hit.payload.get("doc_id", ""),
                "content": hit.payload.get("content", ""),
                "source_file": hit.payload.get("source_file", ""),
                "page_num": hit.payload.get("page_num"),
                "doc_type": hit.payload.get("doc_type", "unknown"),
                "relevance_score": hit.score,
                "equipment_tags": hit.payload.get("equipment_tags", []),
            }
            for hit in results
        ]

    async def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value,
            }
        except Exception:
            return {"points_count": 0, "vectors_count": 0, "status": "not_found"}
