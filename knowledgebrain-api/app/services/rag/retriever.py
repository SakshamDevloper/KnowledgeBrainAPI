"""
KnowledgeBrain — Retriever Service
Hybrid retrieval: semantic search + keyword filtering + re-ranking.
"""

import re
import time
from typing import Optional

from app.services.rag.embedder import embed_text, QdrantService
from app.models.document import DocumentSearchResult, DocType
from app.utils.logging import get_logger

logger = get_logger("retriever")

# Equipment tag regex for query analysis
EQUIPMENT_TAG_RE = re.compile(r'\b([A-Z]{1,4}-[0-9]{3,4}[A-Z]?)\b')


def extract_equipment_tags_from_query(query: str) -> list[str]:
    """Extract equipment tag mentions from a query string."""
    return list(set(EQUIPMENT_TAG_RE.findall(query.upper())))


async def hybrid_retrieve(
    query: str,
    qdrant_service: QdrantService,
    top_k: int = 10,
    equipment_filter: Optional[list[str]] = None,
    doc_type_filter: Optional[list[str]] = None,
) -> list[DocumentSearchResult]:
    """
    Perform hybrid retrieval:
    1. Semantic search via embedding similarity
    2. Keyword filter if query contains equipment tags
    3. Combine, deduplicate, and re-rank
    """
    start_time = time.time()

    # Detect equipment tags in the query
    query_tags = extract_equipment_tags_from_query(query)
    if query_tags and not equipment_filter:
        equipment_filter = query_tags
        logger.info(f"Auto-detected equipment tags in query: {query_tags}")

    # Step 1: Embed the query
    query_embedding = await embed_text(query)

    # Step 2: Semantic search (with optional filters)
    semantic_results = await qdrant_service.search(
        query_embedding=query_embedding,
        top_k=top_k,
        equipment_filter=equipment_filter,
        doc_type_filter=doc_type_filter,
    )

    # Step 3: If we have equipment tags, also do an unfiltered search to catch broader context
    all_results = list(semantic_results)
    if equipment_filter:
        unfiltered_results = await qdrant_service.search(
            query_embedding=query_embedding,
            top_k=top_k // 2,
        )
        all_results.extend(unfiltered_results)

    # Step 4: Deduplicate by chunk_id
    seen_ids = set()
    unique_results = []
    for result in all_results:
        chunk_id = result["chunk_id"]
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_results.append(result)

    # Step 5: Re-rank by relevance score
    unique_results.sort(key=lambda r: r["relevance_score"], reverse=True)

    # Step 6: Take top-k
    top_results = unique_results[:top_k]

    # Convert to DocumentSearchResult
    search_results = [
        DocumentSearchResult(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            content=r["content"],
            source_file=r["source_file"],
            page_num=r.get("page_num"),
            doc_type=DocType(r.get("doc_type", "unknown")),
            relevance_score=r["relevance_score"],
        )
        for r in top_results
    ]

    elapsed = (time.time() - start_time) * 1000
    logger.info(
        f"Hybrid retrieval: {len(search_results)} results in {elapsed:.0f}ms "
        f"(equipment_filter={equipment_filter})"
    )

    return search_results
