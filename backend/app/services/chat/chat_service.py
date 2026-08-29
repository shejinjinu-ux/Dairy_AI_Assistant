"""
Master Multilingual Chat Orchestrator Service
Coordinates Language Detection, Intent Classification, Module Routing, AI Generation, and Session Memory
"""

import logging
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.app.schemas.chat import ChatRequest, ChatResponse, ChatErrorResponse, ChatErrorDetail
from backend.app.services.chat.language_service import language_service
from backend.app.services.chat.intent_service import intent_service
from backend.app.services.chat.silage_chat_service import silage_chat_service
from backend.app.services.chat.nutrition_service import nutrition_service
from backend.app.services.chat.ai_service import ai_service
from backend.app.services.chat.session_service import session_service

from backend.app.core.ownership_guard import ownership_guard
from backend.app.db.farm_cattle_repository import get_farm_cattle_repository

logger = logging.getLogger("dairy_ai.chat.chat_service")


class ChatOrchestratorService:
    """Production Chat Orchestrator Service."""

    def process_message(self, request: ChatRequest, http_request: Optional[Any] = None) -> ChatResponse:
        """
        Processes a multilingual farmer chat message through the complete pipeline.
        Resolves authorized selected animal context and injects authorized analysis history.
        """
        # 1. Message Validation
        clean_message = request.message.strip()
        if not clean_message:
            return ChatResponse(
                success=False,
                reply="Please provide a non-empty message.",
                language="en",
                detected_language="en",
                intent="unknown",
                module="chat",
                session_id=request.session_id or "sess_default"
            )

        # 2. Language Detection & Selection
        target_lang, detected_lang = language_service.resolve_effective_language(
            explicit_language=request.language,
            text=clean_message
        )

        # 3. Derive Authenticated Identity & Validate Ownership
        user_id = request.user_id
        if http_request is not None:
            extracted_uid = ownership_guard.extract_authenticated_user_id(http_request)
            if extracted_uid:
                user_id = extracted_uid

        # 4. Session & History Management (Enforcing user_id ownership)
        session = session_service.get_or_create_session(
            session_id=request.session_id,
            user_id=user_id,
            language=target_lang
        )
        recent_history = session_service.get_history(session.id)
        history_dicts = [m.model_dump() for m in recent_history]

        # 5. Selected Animal Context Resolution (ZERO Fallback Policy)
        selected_cattle = None
        latest_analysis_records = []
        if request.selected_animal_id and user_id:
            repo = get_farm_cattle_repository()
            selected_cattle = repo.get_cattle(
                animal_id=request.selected_animal_id,
                user_id=user_id,
                farm_id=request.farm_id
            )
            if selected_cattle:
                latest_analysis_records = repo.get_latest_analysis(
                    user_id=user_id,
                    animal_id=request.selected_animal_id,
                    limit=5
                )

        # 6. Intent Classification
        intent_match = intent_service.classify(clean_message, conversation_history=history_dicts)
        intent = intent_match.intent
        module = intent_match.module

        # 7. Module Routing & Evaluation
        silage_data = None
        nutrition_data = None
        extracted_metadata: Dict[str, Any] = {
            "intent_confidence": intent_match.confidence,
            "matched_keywords": intent_match.matched_keywords,
            "selected_animal_id": request.selected_animal_id,
            "selected_animal_active": selected_cattle is not None
        }

        # A. Silage Quality Route
        if intent == "silage_quality" or module == "silage":
            is_evaluated, silage_eval, extracted_params = silage_chat_service.evaluate_silage_query(
                clean_message, conversation_history=history_dicts
            )
            silage_data = silage_eval
            extracted_metadata["silage_parameters"] = extracted_params
            extracted_metadata["silage_evaluated"] = is_evaluated

        # B. Nutrition Route
        elif intent == "nutrition" or module == "nutrition":
            nutrition_data = nutrition_service.generate_ration_advisory(
                clean_message,
                language=target_lang,
                conversation_history=history_dicts,
                selected_cattle=selected_cattle,
                analysis_records=latest_analysis_records
            )
            extracted_metadata["nutrition_extracted"] = nutrition_data.extracted_parameters
            extracted_metadata["nutrition_missing"] = nutrition_data.missing_critical_parameters
            extracted_metadata["nutrition_model_active"] = nutrition_data.is_model_predicted

        # 8. AI Response Generation
        reply = ai_service.generate_response(
            user_message=clean_message,
            target_language=target_lang,
            intent=intent,
            module=module,
            conversation_history=history_dicts,
            silage_data=silage_data,
            nutrition_data=nutrition_data,
            selected_cattle=selected_cattle,
            analysis_records=latest_analysis_records
        )

        # 7. Record interaction in session history
        session_service.record_interaction(
            session_id=session.id,
            user_message=clean_message,
            assistant_reply=reply,
            language=target_lang,
            intent=intent,
            module=module
        )

        return ChatResponse(
            success=True,
            reply=reply,
            language=target_lang,
            detected_language=detected_lang,
            intent=intent,
            module=module,
            session_id=session.id,
            metadata=extracted_metadata
        )


# Global singleton orchestrator instance
chat_service = ChatOrchestratorService()
