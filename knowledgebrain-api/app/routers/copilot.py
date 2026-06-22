"""
KnowledgeBrain — Copilot Router
Conversational AI copilot with multi-turn memory, multilingual support, and SSE streaming.
"""

import json
import time
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models.copilot import (
    CopilotRequest,
    CopilotResponse,
    CopilotSource,
    ConversationTurn,
)
from app.utils.logging import get_logger

logger = get_logger("router.copilot")
router = APIRouter(prefix="/copilot", tags=["Copilot"])

# In-memory conversation store (Redis replacement for prototype)
_conversations: dict[str, list[ConversationTurn]] = {}

COPILOT_SYSTEM_PROMPT = """You are KnowledgeBrain, an expert AI assistant for industrial operations at a petroleum refinery. You have access to all plant documents, maintenance records, and regulatory procedures.

Always:
1. Cite sources with [Source: filename, Page X]
2. Flag safety-critical information with [SAFETY]
3. Recommend escalation if you detect an active incident
4. Provide a confidence score (HIGH/MEDIUM/LOW) at the end

Never guess — if you don't know, say so clearly.
When providing maintenance or operational advice, always reference the specific OEM manual or SOP.
For regulatory questions, cite the specific OISD/PESO/Factory Act section."""


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
}


async def translate_text(
    text: str,
    target_language: str,
    anthropic_client,
    model: str,
    preserve_technical: bool = True,
) -> str:
    """Translate text using Claude, preserving technical terms."""
    if target_language == "en":
        return text

    lang_name = LANGUAGE_NAMES.get(target_language, target_language)
    preserve_note = ""
    if preserve_technical:
        preserve_note = (
            " IMPORTANT: Keep all equipment tags (like P-101A, V-204B), "
            "OISD codes, technical measurements, and acronyms in their original English form. "
            "Only translate the natural language parts."
        )

    try:
        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": f"Translate the following text to {lang_name}.{preserve_note}\n\nText:\n{text}",
            }],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


def get_conversation_history(conversation_id: str, max_turns: int = 5) -> list[dict]:
    """Get recent conversation turns for context."""
    turns = _conversations.get(conversation_id, [])
    recent = turns[-max_turns:] if len(turns) > max_turns else turns
    return [{"role": t.role, "content": t.content} for t in recent]


def store_conversation_turn(conversation_id: str, role: str, content: str, language: str = "en"):
    """Store a conversation turn."""
    if conversation_id not in _conversations:
        _conversations[conversation_id] = []
    _conversations[conversation_id].append(
        ConversationTurn(role=role, content=content, language=language)
    )
    # Limit stored turns to 50
    if len(_conversations[conversation_id]) > 50:
        _conversations[conversation_id] = _conversations[conversation_id][-50:]


@router.post("/chat", response_model=CopilotResponse)
async def copilot_chat(request: CopilotRequest):
    """
    Conversational AI copilot endpoint.

    Features:
    - Multi-turn conversation with memory
    - Multilingual support (EN, Hindi, Telugu, Tamil)
    - RAG-powered answers with source citations
    - Neo4j equipment context when tags are provided
    - Safety flag detection
    - Follow-up question suggestions
    """
    settings = get_settings()
    start_time = time.time()

    # Generate or use existing conversation ID
    conversation_id = request.conversation_id or str(uuid4())

    # Get Anthropic client
    anthropic_client = None
    if settings.anthropic_api_key:
        import anthropic
        anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Step 1: Translate if needed
    query_en = request.message
    if request.language != "en" and anthropic_client:
        query_en = await translate_text(
            request.message, "en", anthropic_client, settings.anthropic_model
        )
        logger.info(f"Translated query from {request.language}: {query_en[:100]}")

    # Step 2: RAG retrieval
    sources: list[CopilotSource] = []
    context_text = ""
    try:
        from app.services.rag.embedder import QdrantService
        from app.services.rag.retriever import hybrid_retrieve

        qdrant = QdrantService()
        chunks = await hybrid_retrieve(
            query=query_en,
            qdrant_service=qdrant,
            top_k=settings.default_top_k,
            equipment_filter=request.equipment_context or None,
        )

        for chunk in chunks[:5]:
            page_info = f", Page {chunk.page_num}" if chunk.page_num else ""
            context_text += f"[Source: {chunk.source_file}{page_info}]\n{chunk.content}\n\n---\n\n"
            sources.append(CopilotSource(
                filename=chunk.source_file,
                page=chunk.page_num,
                excerpt=chunk.content[:200],
                relevance=chunk.relevance_score,
            ))
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")

    # Step 3: Neo4j equipment context
    equipment_context_text = ""
    if request.equipment_context:
        try:
            from app.services.graph.neo4j_service import Neo4jService
            neo4j = Neo4jService()
            await neo4j.connect()
            for tag in request.equipment_context[:3]:  # Limit to 3 tags
                ctx = await neo4j.get_equipment_context(tag)
                if ctx.get("equipment"):
                    equipment_context_text += f"\nEquipment {tag}:\n"
                    equipment_context_text += f"  Work Orders: {len(ctx.get('work_orders', []))}\n"
                    equipment_context_text += f"  Incidents: {len(ctx.get('incidents', []))}\n"
                    equipment_context_text += f"  Documents: {len(ctx.get('documents', []))}\n"
                    for wo in ctx.get("work_orders", [])[:3]:
                        equipment_context_text += f"  - WO: {wo}\n"
            await neo4j.close()
        except Exception as e:
            logger.warning(f"Neo4j context fetch failed: {e}")

    # Step 4: Build conversation messages
    history = get_conversation_history(conversation_id)
    messages = []
    for turn in history:
        messages.append(turn)

    # Build the user message with all context
    full_user_msg = ""
    if context_text:
        full_user_msg += f"Context Documents:\n{context_text}\n"
    if equipment_context_text:
        full_user_msg += f"Equipment Context:\n{equipment_context_text}\n"
    full_user_msg += f"Question: {query_en}"

    messages.append({"role": "user", "content": full_user_msg})

    # Step 5: Generate answer
    answer = ""
    confidence = "MEDIUM"
    safety_flag = False
    follow_ups = []

    if anthropic_client:
        try:
            response = await anthropic_client.messages.create(
                model=settings.anthropic_model,
                max_tokens=2000,
                temperature=0.1,
                system=COPILOT_SYSTEM_PROMPT,
                messages=messages,
            )
            answer = response.content[0].text.strip()

            # Detect safety flag
            safety_flag = "[SAFETY]" in answer

            # Parse confidence
            answer_lower = answer.lower()
            if "high" in answer_lower.split("confidence")[-1] if "confidence" in answer_lower else "":
                confidence = "HIGH"
            elif "low" in answer_lower.split("confidence")[-1] if "confidence" in answer_lower else "":
                confidence = "LOW"

            # Generate follow-up suggestions
            follow_up_response = await anthropic_client.messages.create(
                model=settings.anthropic_model,
                max_tokens=200,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Based on this Q&A, suggest 3 concise follow-up questions the user might ask. "
                        f"Return as a JSON array of strings.\n\n"
                        f"Question: {query_en}\nAnswer: {answer[:500]}"
                    ),
                }],
            )
            import re
            follow_text = follow_up_response.content[0].text.strip()
            try:
                if "[" in follow_text:
                    json_match = re.search(r'\[.*\]', follow_text, re.DOTALL)
                    if json_match:
                        follow_ups = json.loads(json_match.group())[:3]
            except Exception:
                follow_ups = []

        except Exception as e:
            logger.error(f"Copilot generation failed: {e}")
            answer = f"I encountered an error generating the response: {str(e)}"
            confidence = "LOW"
    else:
        answer = (
            "Anthropic API key not configured. Here are the relevant document excerpts:\n\n"
            + context_text[:1000] if context_text else "No relevant documents found."
        )
        confidence = "LOW"

    # Step 6: Translate back if needed
    translated_answer = None
    if request.language != "en" and anthropic_client and answer:
        translated_answer = await translate_text(
            answer, request.language, anthropic_client, settings.anthropic_model
        )

    # Step 7: Store conversation turn
    store_conversation_turn(conversation_id, "user", query_en, request.language)
    store_conversation_turn(conversation_id, "assistant", answer, "en")

    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Copilot response in {elapsed:.0f}ms (confidence: {confidence}, safety: {safety_flag})")

    return CopilotResponse(
        answer=answer,
        translated_answer=translated_answer,
        sources=sources,
        confidence=confidence,
        follow_up_suggestions=follow_ups,
        safety_flag=safety_flag,
        conversation_id=conversation_id,
    )


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history by ID."""
    turns = _conversations.get(conversation_id, [])
    return {
        "conversation_id": conversation_id,
        "turns": [{"role": t.role, "content": t.content, "language": t.language} for t in turns],
        "turn_count": len(turns),
    }
