"""
KnowledgeBrain — PDF Processor
Extracts, chunks, and enriches PDF documents for the knowledge base.
"""

import io
import time
from typing import Optional
from uuid import uuid4
from datetime import datetime

import pdfplumber

from app.config import get_settings
from app.models.document import (
    DocType,
    DocumentChunk,
    DocumentMetadata,
    IngestResponse,
)
from app.services.ingestion.entity_extractor import (
    extract_all_entities,
    count_entities_by_type,
)
from app.utils.logging import get_logger

logger = get_logger("pdf_processor")


def recursive_character_split(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """
    Split text into chunks using recursive character splitting.
    Tries to split on paragraph breaks first, then sentences, then words.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    chunks: list[str] = []

    if len(text) <= chunk_size:
        if text.strip():
            chunks.append(text.strip())
        return chunks

    # Find the best separator
    separator = separators[-1]
    for sep in separators:
        if sep in text:
            separator = sep
            break

    splits = text.split(separator) if separator else list(text)

    current_chunk = ""
    for split in splits:
        piece = split if not separator else split + separator
        if len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Overlap: take the tail of the current chunk
            if chunk_overlap > 0 and current_chunk:
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + piece
            else:
                current_chunk = piece

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    """
    Extract text from PDF bytes, returning per-page content.
    Returns: [{page_num: int, text: str}, ...]
    """
    pages = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page_num": i + 1, "text": text})
            logger.info(f"Extracted {len(pages)} pages with text from PDF")
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise ValueError(f"Failed to process PDF: {str(e)}")

    return pages


async def generate_document_summary(
    full_text: str,
    filename: str,
    anthropic_client,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Generate a concise summary of the document using Claude."""
    if not anthropic_client:
        return "Summary generation unavailable (no API key configured)."

    truncated = full_text[:6000]  # Stay within token limits

    try:
        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this industrial document in 3-5 sentences. "
                    f"Focus on: equipment covered, key procedures, safety requirements, "
                    f"and regulatory references.\n\n"
                    f"Document: {filename}\n\n{truncated}"
                ),
            }],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"Summary generation failed: {str(e)}"


async def process_pdf(
    file_bytes: bytes,
    filename: str,
    doc_type: DocType = DocType.PDF,
    anthropic_client=None,
    qdrant_service=None,
    graph_builder=None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IngestResponse:
    """
    Full PDF processing pipeline:
    1. Extract text per page
    2. Chunk with recursive splitting
    3. Extract entities (regex + LLM)
    4. Generate document summary
    5. Store in Qdrant
    6. Build graph nodes (if graph_builder provided)
    """
    start_time = time.time()
    settings = get_settings()
    doc_id = str(uuid4())
    all_chunks: list[DocumentChunk] = []
    all_entities_flat = []

    # Step 1: Extract text from PDF
    pages = extract_text_from_pdf(file_bytes)
    if not pages:
        raise ValueError("No text could be extracted from the PDF. It may be scanned/image-only.")

    full_text = "\n\n".join(p["text"] for p in pages)

    # Step 2 & 3: Chunk each page and extract entities
    chunk_index = 0
    for page_data in pages:
        page_chunks = recursive_character_split(
            page_data["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_text in page_chunks:
            # Extract entities from this chunk
            entities = await extract_all_entities(
                chunk_text,
                anthropic_client=anthropic_client,
                model=settings.anthropic_model,
                use_llm=(anthropic_client is not None),
            )
            all_entities_flat.extend(entities)

            chunk = DocumentChunk(
                doc_id=doc_id,
                content=chunk_text,
                page_num=page_data["page_num"],
                chunk_index=chunk_index,
                source_file=filename,
                doc_type=doc_type,
                entities=entities,
                metadata={
                    "equipment_tags": [
                        e.value for e in entities if e.entity_type == "equipment_tag"
                    ],
                    "regulatory_refs": [
                        e.value for e in entities if e.entity_type == "regulatory_ref"
                    ],
                },
                created_at=datetime.utcnow(),
            )
            all_chunks.append(chunk)
            chunk_index += 1

    # Step 4: Generate document summary
    summary = await generate_document_summary(
        full_text, filename, anthropic_client, settings.anthropic_model
    )

    # Step 5: Store in Qdrant (if service available)
    if qdrant_service is not None:
        await qdrant_service.store_chunks(all_chunks)
        logger.info(f"Stored {len(all_chunks)} chunks in Qdrant")

    # Step 6: Build knowledge graph nodes (if available)
    if graph_builder is not None:
        try:
            await graph_builder.build_from_document(doc_id, filename, doc_type, all_entities_flat)
            logger.info(f"Built graph nodes for document {doc_id}")
        except Exception as e:
            logger.warning(f"Graph building failed (non-fatal): {e}")

    processing_time = (time.time() - start_time) * 1000
    entity_counts = count_entities_by_type(all_entities_flat)

    logger.info(
        f"Processed PDF '{filename}': {len(all_chunks)} chunks, "
        f"{len(all_entities_flat)} entities in {processing_time:.0f}ms"
    )

    return IngestResponse(
        doc_id=doc_id,
        filename=filename,
        doc_type=doc_type,
        chunk_count=len(all_chunks),
        entities_found=entity_counts,
        summary=summary,
        processing_time_ms=round(processing_time, 2),
    )
