"""
KnowledgeBrain — Email Processor
Ingests .eml files and plain-text email archives.
"""

import email
import io
import re
import time
from typing import Optional
from uuid import uuid4
from datetime import datetime
from email import policy

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

logger = get_logger("email_processor")

# Boilerplate footer patterns to filter out
BOILERPLATE_PATTERNS = [
    re.compile(r'^Sent from.*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^This email.*confidential.*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^CONFIDENTIAL.*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^Disclaimer:.*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^This message.*intended.*recipient.*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^If you.*received.*error.*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'_{10,}', re.MULTILINE),
    re.compile(r'-{10,}', re.MULTILINE),
]


def clean_email_body(body: str) -> str:
    """Remove boilerplate footers from email body."""
    cleaned = body
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Remove excessive blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def parse_eml(file_bytes: bytes) -> dict:
    """Parse a .eml file and extract metadata + body."""
    msg = email.message_from_bytes(file_bytes, policy=policy.default)

    # Extract headers
    sender = str(msg.get("From", ""))
    recipients = str(msg.get("To", ""))
    cc = str(msg.get("Cc", ""))
    date = str(msg.get("Date", ""))
    subject = str(msg.get("Subject", ""))
    message_id = str(msg.get("Message-ID", ""))

    # Extract body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                try:
                    body = part.get_payload(decode=True).decode(charset, errors="replace")
                except Exception:
                    body = str(part.get_payload())
                break
    else:
        charset = msg.get_content_charset() or "utf-8"
        try:
            body = msg.get_payload(decode=True).decode(charset, errors="replace")
        except Exception:
            body = str(msg.get_payload())

    body = clean_email_body(body)

    return {
        "sender": sender,
        "recipients": recipients,
        "cc": cc,
        "date": date,
        "subject": subject,
        "message_id": message_id,
        "body": body,
    }


def parse_plain_text_email(text: str) -> dict:
    """Parse a plain text email (e.g., from a .txt export)."""
    lines = text.strip().split("\n")

    sender = ""
    recipients = ""
    date = ""
    subject = ""
    body_start = 0

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if line_lower.startswith("from:"):
            sender = line.split(":", 1)[1].strip()
        elif line_lower.startswith("to:"):
            recipients = line.split(":", 1)[1].strip()
        elif line_lower.startswith("date:"):
            date = line.split(":", 1)[1].strip()
        elif line_lower.startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
        elif line.strip() == "" and i > 0:
            body_start = i + 1
            break

    body = clean_email_body("\n".join(lines[body_start:]))

    return {
        "sender": sender,
        "recipients": recipients,
        "cc": "",
        "date": date,
        "subject": subject,
        "message_id": "",
        "body": body,
    }


async def process_email(
    file_bytes: bytes,
    filename: str,
    doc_type: DocType = DocType.EMAIL,
    anthropic_client=None,
    qdrant_service=None,
    graph_builder=None,
) -> IngestResponse:
    """
    Process an email file (.eml or .txt):
    1. Parse email metadata and body
    2. Clean boilerplate
    3. Extract entities
    4. Store as document chunk
    """
    start_time = time.time()
    settings = get_settings()
    doc_id = str(uuid4())

    # Step 1: Parse email
    if filename.lower().endswith(".eml"):
        email_data = parse_eml(file_bytes)
    else:
        text = file_bytes.decode("utf-8", errors="replace")
        email_data = parse_plain_text_email(text)

    # Step 2: Build chunk content
    header_text = (
        f"From: {email_data['sender']}\n"
        f"To: {email_data['recipients']}\n"
        f"Date: {email_data['date']}\n"
        f"Subject: {email_data['subject']}\n\n"
    )
    full_text = header_text + email_data["body"]

    if not email_data["body"].strip():
        raise ValueError("Email body is empty after cleaning")

    # Step 3: Extract entities
    entities = await extract_all_entities(
        full_text,
        anthropic_client=anthropic_client,
        model=settings.anthropic_model,
        use_llm=(anthropic_client is not None),
    )

    # Step 4: Create chunk
    chunk = DocumentChunk(
        doc_id=doc_id,
        content=full_text,
        chunk_index=0,
        source_file=filename,
        doc_type=doc_type,
        entities=entities,
        metadata={
            "sender": email_data["sender"],
            "recipients": email_data["recipients"],
            "subject": email_data["subject"],
            "date": email_data["date"],
            "equipment_tags": [
                e.value for e in entities if e.entity_type == "equipment_tag"
            ],
            "regulatory_refs": [
                e.value for e in entities if e.entity_type == "regulatory_ref"
            ],
        },
        created_at=datetime.utcnow(),
    )

    # Step 5: Store
    if qdrant_service is not None:
        await qdrant_service.store_chunks([chunk])

    if graph_builder is not None:
        try:
            await graph_builder.build_from_document(doc_id, filename, doc_type, entities)
        except Exception as e:
            logger.warning(f"Graph building failed (non-fatal): {e}")

    processing_time = (time.time() - start_time) * 1000
    entity_counts = count_entities_by_type(entities)

    summary = (
        f"Email from {email_data['sender']} — "
        f"Subject: {email_data['subject']} — "
        f"Date: {email_data['date']}"
    )

    return IngestResponse(
        doc_id=doc_id,
        filename=filename,
        doc_type=doc_type,
        chunk_count=1,
        entities_found=entity_counts,
        summary=summary,
        processing_time_ms=round(processing_time, 2),
    )
