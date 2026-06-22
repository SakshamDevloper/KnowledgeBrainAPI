"""
KnowledgeBrain — Synthesizer Service
Claude-powered answer synthesis from retrieved context chunks.
"""

import re
import time
from typing import Optional

from app.config import get_settings
from app.models.query import QueryResponse, SourceReference
from app.models.document import DocumentSearchResult
from app.utils.logging import get_logger

logger = get_logger("synthesizer")

SYSTEM_PROMPT = """You are an expert industrial knowledge assistant. Answer questions using ONLY the provided context documents. Always cite your sources with [Source: filename, Page X]. If the answer is not in the context, say 'I could not find this information in the available documents.' End every answer with a confidence assessment.

Rules:
1. Be precise and technical — this is for industrial operations professionals.
2. Always cite the specific document and page where you found the information.
3. If information from multiple sources conflicts, note the discrepancy.
4. Flag any safety-critical information with [SAFETY].
5. If the question relates to equipment maintenance, include any relevant thresholds or limits from OEM manuals.
6. Never fabricate information. If unsure, say so clearly."""


def build_context_prompt(chunks: list[DocumentSearchResult], max_chunks: int = 5) -> str:
    """Build the context section of the prompt from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        page_info = f", Page {chunk.page_num}" if chunk.page_num else ""
        context_parts.append(
            f"[Source: {chunk.source_file}{page_info}]\n{chunk.content}\n"
        )
    return "\n---\n".join(context_parts)


def parse_confidence(answer: str) -> str:
    """Extract confidence level from the answer text."""
    answer_lower = answer.lower()
    if "high confidence" in answer_lower or "confidence: high" in answer_lower:
        return "HIGH"
    elif "low confidence" in answer_lower or "confidence: low" in answer_lower:
        return "LOW"
    elif "could not find" in answer_lower or "not available" in answer_lower:
        return "LOW"
    return "MEDIUM"


async def synthesize_answer(
    query: str,
    chunks: list[DocumentSearchResult],
    anthropic_client,
    model: Optional[str] = None,
    max_context_chunks: int = 5,
) -> QueryResponse:
    """
    Generate an answer from retrieved chunks using Claude.

    1. Build context from top chunks
    2. Call Claude with system prompt + context + query
    3. Parse response and extract confidence
    4. Return structured response
    """
    start_time = time.time()
    settings = get_settings()
    model = model or settings.anthropic_model

    if not chunks:
        return QueryResponse(
            answer="I could not find any relevant documents to answer this question. Please try rephrasing your query or check if the relevant documents have been ingested.",
            sources=[],
            confidence="LOW",
            query_time_ms=0,
            chunks_retrieved=0,
        )

    # Build context
    context = build_context_prompt(chunks, max_context_chunks)

    user_message = f"""Context Documents:
{context}

Question: {query}

Provide a detailed, well-cited answer based on the context documents above. End with your confidence level (HIGH/MEDIUM/LOW)."""

    if not anthropic_client:
        # Fallback: return raw chunks if no API key
        combined = "\n\n".join(
            f"[{c.source_file}, Page {c.page_num}]: {c.content[:300]}..."
            for c in chunks[:max_context_chunks]
        )
        elapsed = (time.time() - start_time) * 1000
        return QueryResponse(
            answer=f"[API key not configured — showing raw retrieved context]\n\n{combined}",
            sources=[
                SourceReference(
                    filename=c.source_file,
                    page=c.page_num,
                    excerpt=c.content[:200],
                    relevance_score=c.relevance_score,
                    doc_type=c.doc_type.value,
                )
                for c in chunks[:max_context_chunks]
            ],
            confidence="LOW",
            query_time_ms=round(elapsed, 2),
            chunks_retrieved=len(chunks),
        )

    try:
        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = response.content[0].text.strip()
    except Exception as e:
        logger.error(f"Claude synthesis failed: {e}")
        answer = f"An error occurred while generating the answer: {str(e)}"

    # Parse confidence
    confidence = parse_confidence(answer)

    # Build source references
    sources = [
        SourceReference(
            filename=chunk.source_file,
            page=chunk.page_num,
            excerpt=chunk.content[:200],
            relevance_score=chunk.relevance_score,
            doc_type=chunk.doc_type.value,
        )
        for chunk in chunks[:max_context_chunks]
    ]

    elapsed = (time.time() - start_time) * 1000

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        query_time_ms=round(elapsed, 2),
        chunks_retrieved=len(chunks),
    )
