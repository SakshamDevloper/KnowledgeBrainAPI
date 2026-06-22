"""
KnowledgeBrain — Copilot Models
Pydantic schemas for the conversational AI copilot.
"""

from typing import Optional
from pydantic import BaseModel, Field


class CopilotRequest(BaseModel):
    """Request body for copilot chat."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    conversation_id: Optional[str] = Field(
        default=None, description="Existing conversation ID for multi-turn"
    )
    user_id: str = Field(default="anonymous")
    language: str = Field(
        default="en",
        description="Response language: en, hi, te, ta"
    )
    equipment_context: list[str] = Field(
        default_factory=list,
        description="Equipment tags to focus queries on"
    )


class CopilotSource(BaseModel):
    """Source reference in copilot response."""
    filename: str
    page: Optional[int] = None
    excerpt: str = ""
    relevance: float = 0.0


class CopilotResponse(BaseModel):
    """Response from copilot chat."""
    answer: str
    translated_answer: Optional[str] = Field(
        default=None, description="Translated answer if language != en"
    )
    sources: list[CopilotSource] = Field(default_factory=list)
    confidence: str = Field(default="MEDIUM", description="LOW | MEDIUM | HIGH")
    follow_up_suggestions: list[str] = Field(
        default_factory=list, max_length=3,
        description="Suggested next questions"
    )
    safety_flag: bool = Field(default=False, description="True if safety-critical info detected")
    conversation_id: str = ""


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    role: str = Field(..., description="user or assistant")
    content: str
    language: str = "en"


# ── Maintenance Models ─────────────────────────────

class PredictiveMaintenanceRequest(BaseModel):
    """Request for predictive maintenance analysis."""
    equipment_tag: str = Field(..., description="Equipment identifier e.g. P-101A")
    current_readings: dict = Field(
        default_factory=dict,
        description="Sensor readings: {vibration_mm_s, temperature_c, pressure_bar}"
    )


class PredictiveMaintenanceResponse(BaseModel):
    """Response from predictive maintenance agent."""
    equipment_tag: str
    risk_level: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    recommended_action: str
    urgency: str = Field(description="schedule | soon | immediate")
    supporting_evidence: list[str] = Field(default_factory=list)
    estimated_failure_window: str = ""
    similar_incidents: list[dict] = Field(default_factory=list)


class RCARequest(BaseModel):
    """Request for Root Cause Analysis."""
    incident_description: str
    equipment_tag: str
    incident_date: str = ""


class RCAResponse(BaseModel):
    """Response from RCA agent."""
    incident_summary: str
    timeline: list[dict] = Field(default_factory=list)
    similar_past_incidents: list[dict] = Field(default_factory=list)
    probable_root_causes: list[dict] = Field(default_factory=list)
    recommended_corrective_actions: list[str] = Field(default_factory=list)
    prevention_measures: list[str] = Field(default_factory=list)


# ── Compliance Models ──────────────────────────────

class ComplianceGapRequest(BaseModel):
    """Request for compliance gap detection."""
    equipment_tag: Optional[str] = None


class ComplianceGap(BaseModel):
    """A single compliance gap."""
    equipment_tag: str
    regulation_code: str
    gap_type: str
    severity: str
    days_overdue: int = 0
    details: str = ""


class ComplianceGapResponse(BaseModel):
    """Response from compliance gap detection."""
    gaps: list[ComplianceGap] = Field(default_factory=list)
    total_gaps: int = 0
    critical_count: int = 0


class AuditPackageRequest(BaseModel):
    """Request for pre-audit evidence package."""
    audit_type: str = Field(..., description="OISD | Factory | PESO")
    equipment_tags: list[str] = Field(default_factory=list)


class AuditPackageResponse(BaseModel):
    """Response from audit package generator."""
    audit_type: str
    executive_summary: str
    requirements_status: list[dict] = Field(default_factory=list)
    flagged_gaps: list[dict] = Field(default_factory=list)
    document_index: list[dict] = Field(default_factory=list)


# ── Lessons Learned Models ─────────────────────────

class IncidentAnalysisRequest(BaseModel):
    """Request for incident pattern analysis."""
    incident_report_text: str


class IncidentAnalysisResponse(BaseModel):
    """Response from incident analysis."""
    new_incident_summary: str
    similar_past_incidents: list[dict] = Field(default_factory=list)
    key_patterns: list[str] = Field(default_factory=list)
    warning_signs: list[str] = Field(default_factory=list)
    recommended_investigation_focus: list[str] = Field(default_factory=list)


class ExpertKnowledgeRequest(BaseModel):
    """Request for expert knowledge capture."""
    expert_narrative: str
    expert_name: str
    years_experience: int = 0
    equipment_tags: list[str] = Field(default_factory=list)


class ExpertKnowledgeResponse(BaseModel):
    """Response from expert knowledge capture."""
    knowledge_nodes_created: int
    equipment_tags_linked: list[str] = Field(default_factory=list)
    extracted_tips: list[str] = Field(default_factory=list)
    summary: str = ""


class ProactiveAlert(BaseModel):
    """A proactive alert from the lessons engine."""
    alert_id: str
    equipment_tag: str
    matched_incident_ids: list[str] = Field(default_factory=list)
    risk_summary: str
    recommended_immediate_action: str
    severity: str = "MEDIUM"
    similarity_score: float = 0.0
