"""
KnowledgeBrain — Spreadsheet Processor
Handles Excel (.xlsx) and CSV ingestion for maintenance logs and operational data.
"""

import io
import time
from typing import Optional
from uuid import uuid4
from datetime import datetime

import pandas as pd

from app.config import get_settings
from app.models.document import (
    DocType,
    DocumentChunk,
    IngestResponse,
)
from app.services.ingestion.entity_extractor import (
    extract_all_entities,
    count_entities_by_type,
)
from app.utils.logging import get_logger

logger = get_logger("spreadsheet_processor")

# Columns commonly found in maintenance logs
MAINTENANCE_COLUMNS = {
    "equipment_tag", "equipment", "tag", "tag_id", "asset_id",
    "date", "work_date", "completion_date",
    "work_order", "wo_id", "work_order_id", "wo",
    "technician", "assigned_to", "performed_by",
    "failure_code", "fault_code", "failure_mode",
    "description", "work_description", "findings",
    "resolution", "corrective_action", "action_taken",
}


def detect_column_mapping(columns: list[str]) -> dict[str, str]:
    """Map actual column names to standardized maintenance log fields."""
    mapping = {}
    col_lower = {c: c.lower().strip().replace(" ", "_") for c in columns}

    for actual_col, normalized in col_lower.items():
        if normalized in {"equipment_tag", "equipment", "tag", "tag_id", "asset_id", "asset"}:
            mapping["equipment_tag"] = actual_col
        elif normalized in {"date", "work_date", "completion_date", "start_date"}:
            mapping["date"] = actual_col
        elif normalized in {"work_order", "wo_id", "work_order_id", "wo", "wo_number"}:
            mapping["work_order_id"] = actual_col
        elif normalized in {"technician", "assigned_to", "performed_by", "engineer"}:
            mapping["technician"] = actual_col
        elif normalized in {"failure_code", "fault_code", "failure_mode"}:
            mapping["failure_code"] = actual_col
        elif normalized in {"description", "work_description", "findings", "details"}:
            mapping["description"] = actual_col
        elif normalized in {"resolution", "corrective_action", "action_taken", "remedy"}:
            mapping["resolution"] = actual_col

    return mapping


def row_to_text(row: dict, columns: list[str]) -> str:
    """Convert a spreadsheet row to a readable text chunk."""
    parts = []
    for col in columns:
        val = row.get(col)
        if val is not None and str(val).strip() and str(val).lower() != "nan":
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


async def process_spreadsheet(
    file_bytes: bytes,
    filename: str,
    doc_type: DocType = DocType.SPREADSHEET,
    anthropic_client=None,
    qdrant_service=None,
    graph_builder=None,
) -> IngestResponse:
    """
    Process an Excel or CSV file:
    1. Read into DataFrame
    2. Detect maintenance log columns
    3. Convert each row to a text chunk
    4. Extract entities
    5. Generate equipment-level summaries
    6. Store in Qdrant
    """
    start_time = time.time()
    settings = get_settings()
    doc_id = str(uuid4())
    all_chunks: list[DocumentChunk] = []
    all_entities_flat = []

    # Step 1: Read file
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Failed to read spreadsheet: {str(e)}")

    if df.empty:
        raise ValueError("Spreadsheet is empty")

    columns = list(df.columns)
    col_mapping = detect_column_mapping(columns)
    logger.info(f"Detected column mapping: {col_mapping}")

    # Step 2: Process each row
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        chunk_text = row_to_text(row_dict, columns)

        if not chunk_text.strip():
            continue

        # Extract entities
        entities = await extract_all_entities(
            chunk_text,
            anthropic_client=anthropic_client,
            model=settings.anthropic_model,
            use_llm=False,  # Use regex only for spreadsheet rows (faster)
        )
        all_entities_flat.extend(entities)

        # Build metadata from detected columns
        metadata = {"row_index": idx}
        if "equipment_tag" in col_mapping:
            tag_val = str(row_dict.get(col_mapping["equipment_tag"], "")).strip()
            if tag_val and tag_val.lower() != "nan":
                metadata["equipment_tags"] = [tag_val]
        if "work_order_id" in col_mapping:
            wo_val = str(row_dict.get(col_mapping["work_order_id"], "")).strip()
            if wo_val and wo_val.lower() != "nan":
                metadata["work_order_id"] = wo_val

        chunk = DocumentChunk(
            doc_id=doc_id,
            content=chunk_text,
            chunk_index=idx,
            source_file=filename,
            doc_type=doc_type,
            entities=entities,
            metadata=metadata,
            created_at=datetime.utcnow(),
        )
        all_chunks.append(chunk)

    # Step 3: Generate equipment-level summaries
    summary_parts = []
    if "equipment_tag" in col_mapping:
        equipment_groups = df.groupby(col_mapping["equipment_tag"])
        for tag, group_df in equipment_groups:
            summary_parts.append(
                f"Equipment {tag}: {len(group_df)} records"
            )
    summary = f"Spreadsheet '{filename}' with {len(all_chunks)} rows. " + "; ".join(summary_parts[:10])

    # Step 4: Store in Qdrant
    if qdrant_service is not None:
        await qdrant_service.store_chunks(all_chunks)

    # Step 5: Build graph
    if graph_builder is not None:
        try:
            await graph_builder.build_from_document(doc_id, filename, doc_type, all_entities_flat)
        except Exception as e:
            logger.warning(f"Graph building failed (non-fatal): {e}")

    processing_time = (time.time() - start_time) * 1000
    entity_counts = count_entities_by_type(all_entities_flat)

    logger.info(
        f"Processed spreadsheet '{filename}': {len(all_chunks)} rows, "
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
