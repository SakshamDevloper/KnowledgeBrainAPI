"""
KnowledgeBrain — Lessons Learned Router
Incident pattern analysis, proactive alerts, and expert knowledge capture.
"""

import json
from uuid import uuid4
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.copilot import (
    IncidentAnalysisRequest,
    IncidentAnalysisResponse,
    ExpertKnowledgeRequest,
    ExpertKnowledgeResponse,
    ProactiveAlert,
)
from app.utils.logging import get_logger

logger = get_logger("router.lessons")
router = APIRouter(prefix="/lessons", tags=["Lessons Learned"])

# In-memory alert store for prototype
_active_alerts: list[ProactiveAlert] = []


@router.post("/analyze-incident", response_model=IncidentAnalysisResponse)
async def analyze_incident(request: IncidentAnalysisRequest):
    """
    Analyze an incident report:
    1. Extract structured data from incident text
    2. Search for similar historical incidents
    3. Identify patterns and warning signs
    4. Generate investigation recommendations
    """
    settings = get_settings()

    # Search for similar incidents in Neo4j
    similar_incidents = []
    try:
        from app.services.graph.neo4j_service import Neo4jService
        neo4j = Neo4jService()
        await neo4j.connect()
        similar_incidents = await neo4j.find_similar_incidents(request.incident_report_text)
        await neo4j.close()
    except Exception as e:
        logger.warning(f"Neo4j unavailable: {e}")

    # Search RAG for related documents
    rag_context = ""
    try:
        from app.services.rag.embedder import QdrantService
        from app.services.rag.retriever import hybrid_retrieve

        qdrant = QdrantService()
        chunks = await hybrid_retrieve(
            query=f"incident failure root cause {request.incident_report_text[:200]}",
            qdrant_service=qdrant,
            top_k=5,
        )
        rag_context = "\n".join([c.content for c in chunks[:3]])
    except Exception as e:
        logger.warning(f"RAG unavailable: {e}")

    if not settings.anthropic_api_key:
        return IncidentAnalysisResponse(
            new_incident_summary=request.incident_report_text[:500],
            similar_past_incidents=similar_incidents,
            key_patterns=["AI analysis unavailable"],
            warning_signs=[],
            recommended_investigation_focus=[],
        )

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    prompt = f"""Analyze this incident report for an industrial facility.

New Incident Report:
{request.incident_report_text[:3000]}

Similar Historical Incidents:
{json.dumps(similar_incidents[:5], indent=2, default=str)}

Related Documents:
{rag_context[:2000]}

Provide analysis as JSON:
{{
  "new_incident_summary": "brief summary of the new incident",
  "similar_past_incidents": [{{"date": "...", "description": "...", "root_cause": "...", "resolution": "...", "similarity_score": 0.8}}],
  "key_patterns": ["pattern identified across incidents"],
  "warning_signs": ["early warning signs to watch for"],
  "recommended_investigation_focus": ["area to investigate"]
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
            result = json.loads(json_match.group())
        else:
            result = {"new_incident_summary": result_text}

        return IncidentAnalysisResponse(
            new_incident_summary=result.get("new_incident_summary", ""),
            similar_past_incidents=result.get("similar_past_incidents", []),
            key_patterns=result.get("key_patterns", []),
            warning_signs=result.get("warning_signs", []),
            recommended_investigation_focus=result.get("recommended_investigation_focus", []),
        )
    except Exception as e:
        logger.error(f"Incident analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capture-expert-knowledge", response_model=ExpertKnowledgeResponse)
async def capture_expert_knowledge(request: ExpertKnowledgeRequest):
    """
    Capture and structure expert knowledge:
    1. Extract structured tips and procedures from narrative
    2. Link to relevant equipment nodes in Neo4j
    3. Store as enriched knowledge for the copilot
    """
    settings = get_settings()

    if not settings.anthropic_api_key:
        return ExpertKnowledgeResponse(
            knowledge_nodes_created=0,
            equipment_tags_linked=[],
            extracted_tips=["AI extraction unavailable"],
            summary="API key required for knowledge extraction.",
        )

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    prompt = f"""Extract structured knowledge from this expert's narrative.

Expert: {request.expert_name} ({request.years_experience} years experience)
Equipment Tags Mentioned: {', '.join(request.equipment_tags)}

Narrative:
{request.expert_narrative[:4000]}

Extract as JSON:
{{
  "extracted_tips": ["tip 1 — specific, actionable advice"],
  "failure_patterns": [{{"equipment": "...", "pattern": "...", "prevention": "..."}}],
  "undocumented_procedures": [{{"procedure": "...", "context": "..."}}],
  "summary": "2-3 sentence summary of the key knowledge captured"
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
            result = json.loads(json_match.group())
        else:
            result = {"extracted_tips": [result_text], "summary": result_text[:200]}

        # Store in Neo4j
        nodes_created = 0
        try:
            from app.services.graph.neo4j_service import Neo4jService
            neo4j = Neo4jService()
            await neo4j.connect()

            # Create person node for the expert
            await neo4j.upsert_person(
                name=request.expert_name,
                role="Subject Matter Expert",
                years_experience=request.years_experience,
            )
            nodes_created += 1

            # Store knowledge as document nodes linked to equipment
            from app.services.rag.embedder import QdrantService, embed_text
            from app.models.document import DocumentChunk, DocType
            from datetime import datetime

            qdrant = QdrantService()
            await qdrant.ensure_collection()

            for tip in result.get("extracted_tips", []):
                doc_id = str(uuid4())
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    content=f"Expert Knowledge ({request.expert_name}): {tip}",
                    source_file=f"expert_{request.expert_name.replace(' ', '_')}",
                    doc_type=DocType.UNKNOWN,
                    metadata={
                        "source_type": "expert_knowledge",
                        "expert_name": request.expert_name,
                        "equipment_tags": request.equipment_tags,
                    },
                    created_at=datetime.utcnow(),
                )
                await qdrant.store_chunks([chunk])
                nodes_created += 1

            await neo4j.close()
        except Exception as e:
            logger.warning(f"Knowledge storage failed: {e}")

        return ExpertKnowledgeResponse(
            knowledge_nodes_created=nodes_created,
            equipment_tags_linked=request.equipment_tags,
            extracted_tips=result.get("extracted_tips", []),
            summary=result.get("summary", ""),
        )
    except Exception as e:
        logger.error(f"Expert knowledge capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/proactive-scan")
async def proactive_scan():
    """
    Run a proactive scan for potential failure patterns.
    In production, this would be a scheduled background task.
    """
    # For prototype, return any stored alerts
    return {
        "alerts": [a.model_dump() for a in _active_alerts],
        "scan_status": "completed",
        "alert_count": len(_active_alerts),
    }


@router.get("/alerts")
async def get_alerts():
    """Get all active proactive alerts."""
    return {
        "alerts": [a.model_dump() for a in _active_alerts],
        "total": len(_active_alerts),
    }
