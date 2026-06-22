"""
KnowledgeBrain — Entity Extractor
Hybrid regex + LLM approach for extracting structured entities from document chunks.
"""

import re
import json
from typing import Optional
from app.models.document import ExtractedEntity
from app.utils.logging import get_logger

logger = get_logger("entity_extractor")

# ── Regex Patterns ─────────────────────────────────────

EQUIPMENT_TAG_PATTERN = re.compile(
    r'\b([A-Z]{1,4}-[0-9]{3,4}[A-Z]?)\b'
)

PROCESS_PARAM_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*'
    r'(°C|°F|deg\s*C|deg\s*F|bar|barg|psi|psig|MPa|kPa|'
    r'm³/h|m3/h|kg/h|l/min|GPM|'
    r'kg/cm²|kg/cm2|'
    r'mm/s|mm/sec|in/s|'
    r'RPM|rpm|Hz|'
    r'%|ppm|ppb)',
    re.IGNORECASE
)

REGULATORY_REF_PATTERN = re.compile(
    r'(OISD[- ]?\d{1,3}(?:\.\d+)?|'
    r'OISD[- ]?STD[- ]?\d{1,3}|'
    r'PESO[- ]?\d*|'
    r'IS[- ]\d{3,5}(?::\d{4})?|'
    r'Factory\s+Act.*?Section\s+\d+|'
    r'PNGRB[- ]?\d*|'
    r'NBC[- ]?\d*|'
    r'API[- ]\d{3,4})',
    re.IGNORECASE
)

DATE_PATTERN = re.compile(
    r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|'
    r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})|'
    r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})|'
    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})',
    re.IGNORECASE
)

WORK_ORDER_PATTERN = re.compile(
    r'\b(WO[- ]?\d{4,8}|MWO[- ]?\d{4,8}|PM[- ]?\d{4,8})\b',
    re.IGNORECASE
)


def extract_entities_regex(text: str) -> list[ExtractedEntity]:
    """Extract entities using regex patterns. Fast, high-confidence."""
    entities: list[ExtractedEntity] = []

    # Equipment tags
    for match in EQUIPMENT_TAG_PATTERN.finditer(text):
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        entities.append(ExtractedEntity(
            entity_type="equipment_tag",
            value=match.group(1),
            confidence=0.95,
            source="regex",
            context=text[start:end].strip(),
        ))

    # Process parameters
    for match in PROCESS_PARAM_PATTERN.finditer(text):
        value_str = f"{match.group(1)} {match.group(2)}"
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        entities.append(ExtractedEntity(
            entity_type="process_param",
            value=value_str,
            confidence=0.90,
            source="regex",
            context=text[start:end].strip(),
        ))

    # Regulatory references
    for match in REGULATORY_REF_PATTERN.finditer(text):
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        entities.append(ExtractedEntity(
            entity_type="regulatory_ref",
            value=match.group(0).strip(),
            confidence=0.95,
            source="regex",
            context=text[start:end].strip(),
        ))

    # Dates
    for match in DATE_PATTERN.finditer(text):
        date_val = next(g for g in match.groups() if g is not None)
        entities.append(ExtractedEntity(
            entity_type="date",
            value=date_val.strip(),
            confidence=0.85,
            source="regex",
        ))

    # Work order IDs
    for match in WORK_ORDER_PATTERN.finditer(text):
        entities.append(ExtractedEntity(
            entity_type="document_id",
            value=match.group(1).upper(),
            confidence=0.95,
            source="regex",
        ))

    return entities


async def extract_entities_llm(
    text: str,
    anthropic_client,
    model: str = "claude-sonnet-4-6",
) -> list[ExtractedEntity]:
    """Extract entities using Claude for items regex can't capture (names, roles, etc.)."""
    entities: list[ExtractedEntity] = []

    extraction_prompt = f"""Extract the following from this industrial document text. Return ONLY valid JSON.

TEXT:
{text[:2000]}

Extract:
1. "personnel": list of {{"name": "...", "role": "..."}} — names of people and their job roles
2. "document_ids": list of document IDs, report numbers, or reference codes NOT already matching WO-XXXX patterns
3. "equipment_names": list of {{"tag": "...", "description": "..."}} — equipment described by name but not by tag code

Return JSON:
{{"personnel": [...], "document_ids": [...], "equipment_names": [...]}}

If nothing found for a category, return an empty list."""

    try:
        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": extraction_prompt}],
        )
        result_text = response.content[0].text.strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in result_text:
            json_match = re.search(r'```(?:json)?\s*(.*?)```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1).strip()

        result = json.loads(result_text)

        # Personnel
        for person in result.get("personnel", []):
            if person.get("name"):
                entities.append(ExtractedEntity(
                    entity_type="person",
                    value=f"{person['name']} ({person.get('role', 'unknown')})",
                    confidence=0.80,
                    source="llm",
                ))

        # Document IDs
        for doc_id in result.get("document_ids", []):
            if doc_id:
                entities.append(ExtractedEntity(
                    entity_type="document_id",
                    value=str(doc_id),
                    confidence=0.75,
                    source="llm",
                ))

        # Equipment names
        for equip in result.get("equipment_names", []):
            if equip.get("tag") or equip.get("description"):
                entities.append(ExtractedEntity(
                    entity_type="equipment_tag",
                    value=equip.get("tag", equip.get("description", "")),
                    confidence=0.70,
                    source="llm",
                ))

    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM entity extraction response as JSON")
    except Exception as e:
        logger.error(f"LLM entity extraction failed: {e}")

    return entities


async def extract_all_entities(
    text: str,
    anthropic_client=None,
    model: str = "claude-sonnet-4-6",
    use_llm: bool = True,
) -> list[ExtractedEntity]:
    """
    Run full hybrid extraction: regex first, then LLM for gaps.
    If anthropic_client is None or use_llm is False, only regex extraction runs.
    """
    # Phase 1: Regex (fast, high confidence)
    entities = extract_entities_regex(text)
    logger.debug(f"Regex extracted {len(entities)} entities")

    # Phase 2: LLM (slower, catches what regex misses)
    if use_llm and anthropic_client is not None:
        llm_entities = await extract_entities_llm(text, anthropic_client, model)
        logger.debug(f"LLM extracted {len(llm_entities)} additional entities")

        # Deduplicate: don't add LLM entities that overlap with regex findings
        existing_values = {e.value.upper() for e in entities}
        for entity in llm_entities:
            if entity.value.upper() not in existing_values:
                entities.append(entity)
                existing_values.add(entity.value.upper())

    return entities


def count_entities_by_type(entities: list[ExtractedEntity]) -> dict[str, int]:
    """Count entities grouped by type."""
    counts: dict[str, int] = {}
    for entity in entities:
        counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
    return counts
