"""
KnowledgeBrain — Maintenance Router
Predictive maintenance and Root Cause Analysis endpoints.
"""

import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.copilot import (
    PredictiveMaintenanceRequest,
    PredictiveMaintenanceResponse,
    RCARequest,
    RCAResponse,
)
from app.utils.logging import get_logger

logger = get_logger("router.maintenance")
router = APIRouter(prefix="/maintenance", tags=["Maintenance Intelligence"])


@router.post("/predict", response_model=PredictiveMaintenanceResponse)
async def predictive_maintenance(request: PredictiveMaintenanceRequest):
    """
    Predictive maintenance analysis using multi-step agent workflow:
    1. Fetch equipment history from Neo4j
    2. Fetch OEM thresholds from RAG
    3. Compare current readings against thresholds
    4. Check for historical failure patterns
    5. Generate recommendation via Claude
    """
    settings = get_settings()
    tag = request.equipment_tag
    readings = request.current_readings

    # Step 1: Fetch equipment history
    equipment_context = {}
    try:
        from app.services.graph.neo4j_service import Neo4jService
        neo4j = Neo4jService()
        await neo4j.connect()
        equipment_context = await neo4j.get_equipment_context(tag)
        similar_incidents = await neo4j.find_similar_incidents(
            f"{tag} vibration temperature anomaly failure"
        )
        await neo4j.close()
    except Exception as e:
        logger.warning(f"Neo4j unavailable: {e}")
        similar_incidents = []

    # Step 2: Fetch OEM thresholds from RAG
    oem_context = ""
    try:
        from app.services.rag.embedder import QdrantService
        from app.services.rag.retriever import hybrid_retrieve

        qdrant = QdrantService()
        oem_query = f"OEM manual {tag} vibration threshold temperature limits acceptable range"
        chunks = await hybrid_retrieve(
            query=oem_query,
            qdrant_service=qdrant,
            top_k=5,
            equipment_filter=[tag],
        )
        oem_context = "\n".join([c.content for c in chunks[:3]])
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")

    # Step 3: Generate recommendation via Claude
    anthropic_client = None
    if settings.anthropic_api_key:
        import anthropic
        anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    if not anthropic_client:
        return PredictiveMaintenanceResponse(
            equipment_tag=tag,
            risk_level="MEDIUM",
            recommended_action="Manual review required — AI analysis unavailable (no API key).",
            urgency="schedule",
            supporting_evidence=["AI analysis unavailable"],
            estimated_failure_window="Unknown",
            similar_incidents=[],
        )

    prompt = f"""Analyze the following equipment condition and provide a predictive maintenance recommendation.

Equipment Tag: {tag}
Current Sensor Readings: {json.dumps(readings, indent=2)}

Equipment History (from knowledge graph):
- Work Orders: {len(equipment_context.get('work_orders', []))} records
- Past Incidents: {len(equipment_context.get('incidents', []))} records
- Work order details: {json.dumps(equipment_context.get('work_orders', [])[:5], indent=2, default=str)}
- Incident details: {json.dumps(equipment_context.get('incidents', [])[:3], indent=2, default=str)}

OEM Manual Context:
{oem_context[:2000] if oem_context else "No OEM data available in knowledge base."}

Similar Historical Incidents:
{json.dumps(similar_incidents[:3], indent=2, default=str) if similar_incidents else "None found."}

Provide your analysis as JSON:
{{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "recommended_action": "detailed action description",
  "urgency": "schedule|soon|immediate",
  "supporting_evidence": ["evidence point 1", "evidence point 2"],
  "estimated_failure_window": "estimated time until failure if no action"
}}"""

    try:
        response = await anthropic_client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = response.content[0].text.strip()

        # Parse JSON from response
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"risk_level": "MEDIUM", "recommended_action": result_text, "urgency": "schedule"}

        return PredictiveMaintenanceResponse(
            equipment_tag=tag,
            risk_level=result.get("risk_level", "MEDIUM"),
            recommended_action=result.get("recommended_action", "Review required"),
            urgency=result.get("urgency", "schedule"),
            supporting_evidence=result.get("supporting_evidence", []),
            estimated_failure_window=result.get("estimated_failure_window", "Unknown"),
            similar_incidents=similar_incidents[:3],
        )
    except Exception as e:
        logger.error(f"Predictive maintenance analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rca", response_model=RCAResponse)
async def root_cause_analysis(request: RCARequest):
    """
    Root Cause Analysis using the 5 Whys methodology:
    1. Fetch incident context from Neo4j + RAG
    2. Reconstruct event timeline
    3. Find historical RCA reports
    4. Generate structured RCA report via Claude
    """
    settings = get_settings()

    # Gather context
    equipment_context = {}
    try:
        from app.services.graph.neo4j_service import Neo4jService
        neo4j = Neo4jService()
        await neo4j.connect()
        equipment_context = await neo4j.get_equipment_context(request.equipment_tag)
        similar = await neo4j.find_similar_incidents(request.incident_description)
        await neo4j.close()
    except Exception as e:
        logger.warning(f"Neo4j unavailable: {e}")
        similar = []

    # RAG context
    rag_context = ""
    try:
        from app.services.rag.embedder import QdrantService
        from app.services.rag.retriever import hybrid_retrieve

        qdrant = QdrantService()
        chunks = await hybrid_retrieve(
            query=f"root cause analysis {request.equipment_tag} {request.incident_description[:100]}",
            qdrant_service=qdrant,
            top_k=5,
        )
        rag_context = "\n".join([c.content for c in chunks[:3]])
    except Exception as e:
        logger.warning(f"RAG failed: {e}")

    # Generate RCA
    anthropic_client = None
    if settings.anthropic_api_key:
        import anthropic
        anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    if not anthropic_client:
        return RCAResponse(
            incident_summary=request.incident_description,
            probable_root_causes=[{"cause": "AI analysis unavailable", "confidence": 0, "evidence": "No API key"}],
        )

    prompt = f"""Perform a Root Cause Analysis using the 5 Whys methodology.

Incident Description: {request.incident_description}
Equipment Tag: {request.equipment_tag}
Incident Date: {request.incident_date}

Equipment History:
{json.dumps(equipment_context.get('work_orders', [])[:5], indent=2, default=str)}

Similar Past Incidents:
{json.dumps(similar[:3], indent=2, default=str)}

Related Documents:
{rag_context[:2000]}

Generate a structured RCA report as JSON:
{{
  "incident_summary": "brief summary",
  "timeline": [{{"date": "...", "event": "..."}}],
  "similar_past_incidents": [{{"date": "...", "description": "...", "root_cause": "...", "resolution": "..."}}],
  "probable_root_causes": [{{"cause": "...", "confidence": 0.8, "evidence": "..."}}],
  "recommended_corrective_actions": ["action1", "action2"],
  "prevention_measures": ["measure1", "measure2"]
}}"""

    try:
        response = await anthropic_client.messages.create(
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
            result = {"incident_summary": result_text}

        return RCAResponse(
            incident_summary=result.get("incident_summary", request.incident_description),
            timeline=result.get("timeline", []),
            similar_past_incidents=result.get("similar_past_incidents", []),
            probable_root_causes=result.get("probable_root_causes", []),
            recommended_corrective_actions=result.get("recommended_corrective_actions", []),
            prevention_measures=result.get("prevention_measures", []),
        )
    except Exception as e:
        logger.error(f"RCA generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
