"""
KnowledgeBrain — Compliance Router
Regulatory compliance gap detection and audit package generation.
"""

import json
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.copilot import (
    ComplianceGapRequest,
    ComplianceGapResponse,
    ComplianceGap,
    AuditPackageRequest,
    AuditPackageResponse,
)
from app.utils.logging import get_logger

logger = get_logger("router.compliance")
router = APIRouter(prefix="/compliance", tags=["Compliance Intelligence"])


@router.post("/gaps", response_model=ComplianceGapResponse)
async def detect_compliance_gaps(request: ComplianceGapRequest):
    """
    Detect compliance gaps:
    - Check inspection intervals against regulatory requirements
    - Verify SOP currency
    - Flag overdue inspections
    """
    settings = get_settings()
    gaps: list[ComplianceGap] = []

    # Query Neo4j for equipment and regulation data
    try:
        from app.services.graph.neo4j_service import Neo4jService
        neo4j = Neo4jService()
        await neo4j.connect()

        if request.equipment_tag:
            compliance_data = await neo4j.get_compliance_status(request.equipment_tag)
            for item in compliance_data:
                reg = item.get("regulation", {})
                last_inspection = item.get("last_inspection")
                # Flag as gap if no inspection recorded
                if not last_inspection:
                    gaps.append(ComplianceGap(
                        equipment_tag=request.equipment_tag,
                        regulation_code=reg.get("code", "Unknown"),
                        gap_type="missing_inspection",
                        severity="HIGH",
                        days_overdue=0,
                        details=f"No inspection record found for {reg.get('code', 'Unknown')}",
                    ))
        else:
            # Check all equipment — simplified for prototype
            logger.info("Running compliance gap check for all equipment")

        await neo4j.close()

    except Exception as e:
        logger.warning(f"Neo4j unavailable for compliance check: {e}")

    # Use RAG to find additional compliance information
    try:
        from app.services.rag.embedder import QdrantService
        from app.services.rag.retriever import hybrid_retrieve

        qdrant = QdrantService()
        tag_filter = [request.equipment_tag] if request.equipment_tag else None
        chunks = await hybrid_retrieve(
            query="inspection overdue compliance gap non-conformance",
            qdrant_service=qdrant,
            top_k=5,
            equipment_filter=tag_filter,
        )

        # Use Claude to analyze compliance gaps from documents
        if settings.anthropic_api_key and chunks:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

            context = "\n".join([c.content for c in chunks[:5]])
            prompt = f"""Analyze these documents for compliance gaps. Equipment: {request.equipment_tag or 'all'}

Documents:
{context[:3000]}

Return JSON array of gaps:
[{{"equipment_tag": "...", "regulation_code": "...", "gap_type": "...", "severity": "HIGH|MEDIUM|LOW", "days_overdue": 0, "details": "..."}}]

If no gaps found, return empty array []."""

            response = await client.messages.create(
                model=settings.anthropic_model,
                max_tokens=1500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            import re
            result_text = response.content[0].text.strip()
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                detected_gaps = json.loads(json_match.group())
                for g in detected_gaps:
                    gaps.append(ComplianceGap(**g))

    except Exception as e:
        logger.warning(f"RAG compliance check failed: {e}")

    critical_count = sum(1 for g in gaps if g.severity in ("HIGH", "CRITICAL"))

    return ComplianceGapResponse(
        gaps=gaps,
        total_gaps=len(gaps),
        critical_count=critical_count,
    )


@router.post("/audit-package", response_model=AuditPackageResponse)
async def generate_audit_package(request: AuditPackageRequest):
    """
    Generate a pre-audit compliance evidence package:
    1. Gather all inspection reports, calibration certs, maintenance records
    2. Generate executive summary via Claude
    3. Create requirements vs. status table
    4. Flag gaps with recommended actions
    """
    settings = get_settings()

    # Gather documents
    all_context = ""
    document_index = []

    try:
        from app.services.rag.embedder import QdrantService
        from app.services.rag.retriever import hybrid_retrieve

        qdrant = QdrantService()
        for tag in request.equipment_tags:
            chunks = await hybrid_retrieve(
                query=f"{request.audit_type} inspection calibration maintenance {tag}",
                qdrant_service=qdrant,
                top_k=5,
                equipment_filter=[tag],
            )
            for chunk in chunks:
                all_context += f"[{chunk.source_file}] {chunk.content}\n\n"
                document_index.append({
                    "filename": chunk.source_file,
                    "page": chunk.page_num,
                    "type": chunk.doc_type.value,
                    "equipment_tag": tag,
                })
    except Exception as e:
        logger.warning(f"Document gathering failed: {e}")

    # Generate package with Claude
    if not settings.anthropic_api_key:
        return AuditPackageResponse(
            audit_type=request.audit_type,
            executive_summary="AI analysis unavailable (no API key configured).",
            requirements_status=[],
            flagged_gaps=[],
            document_index=document_index,
        )

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    prompt = f"""Generate a compliance audit evidence package.

Audit Type: {request.audit_type}
Equipment Tags: {', '.join(request.equipment_tags)}

Available Documents:
{all_context[:4000]}

Generate a structured audit package as JSON:
{{
  "executive_summary": "2-3 paragraph summary of compliance status",
  "requirements_status": [
    {{"requirement": "...", "status": "compliant|non-compliant|partial", "evidence": "...", "notes": "..."}}
  ],
  "flagged_gaps": [
    {{"gap": "...", "severity": "HIGH|MEDIUM|LOW", "recommended_action": "..."}}
  ]
}}"""

    try:
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=3000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = response.content[0].text.strip()

        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"executive_summary": result_text}

        return AuditPackageResponse(
            audit_type=request.audit_type,
            executive_summary=result.get("executive_summary", ""),
            requirements_status=result.get("requirements_status", []),
            flagged_gaps=result.get("flagged_gaps", []),
            document_index=document_index,
        )
    except Exception as e:
        logger.error(f"Audit package generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-regulation-change")
async def check_regulation_change(
    new_regulation_text: str,
    regulation_code: str,
):
    """
    Check impact of a regulation change:
    1. Compare new regulation text against existing procedures
    2. Identify affected SOPs
    3. List specific clauses needing updates
    """
    settings = get_settings()

    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="Anthropic API key required")

    # Find procedures referencing this regulation
    try:
        from app.services.rag.embedder import QdrantService
        from app.services.rag.retriever import hybrid_retrieve

        qdrant = QdrantService()
        chunks = await hybrid_retrieve(
            query=f"{regulation_code} procedure SOP",
            qdrant_service=qdrant,
            top_k=10,
        )
        affected_procedures = "\n".join([
            f"[{c.source_file}] {c.content}" for c in chunks[:5]
        ])
    except Exception:
        affected_procedures = ""

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    prompt = f"""A regulation has been updated. Analyze the impact.

Regulation Code: {regulation_code}
New Regulation Text:
{new_regulation_text[:3000]}

Current Procedures Referencing This Regulation:
{affected_procedures[:3000]}

Identify:
1. Which procedures need updating
2. Specific clauses that changed
3. Recommended updates

Return as JSON:
{{
  "affected_procedures": [{{"filename": "...", "specific_changes_needed": "..."}}],
  "changed_clauses": ["clause 1", "clause 2"],
  "change_notification": "draft notification text"
}}"""

    try:
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = response.content[0].text.strip()

        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"analysis": result_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
