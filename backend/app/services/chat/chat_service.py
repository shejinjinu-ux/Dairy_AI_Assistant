"""
Master Multilingual Chat Orchestrator Service
Coordinates Language Detection, Intent Classification, Module Routing, AI Generation,
Persistent Cattle Tag ID Context, and Session Memory.
"""

import re
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
from backend.app.services.vaccination_service import vaccination_service
from backend.app.services.lactation_service import lactation_service

from backend.app.core.ownership_guard import ownership_guard
from backend.app.db.farm_cattle_repository import get_farm_cattle_repository, normalize_tag_id

logger = logging.getLogger("dairy_ai.chat.chat_service")

# Regex to detect cattle Tag ID patterns in user messages (e.g. COW-1001, TAG-ENDP-01, COW_BOB_01)
TAG_ID_PATTERN = re.compile(r'\b((?:COW|TAG|ANIMAL|BUFF|BUFFALO)[\-_A-Za-z0-9]+)\b', re.IGNORECASE)


class ChatOrchestratorService:
    """Production Chat Orchestrator Service with grounded cattle Tag ID resolution."""

    def process_message(self, request: ChatRequest, http_request: Optional[Any] = None) -> ChatResponse:
        """
        Processes a multilingual farmer chat message through the complete pipeline.
        Resolves authorized cattle Tag ID records and injects persistent milk/vaccine/lactation context.
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

        # 5. Tag ID Detection & Cattle Record Resolution
        repo = get_farm_cattle_repository()
        extracted_tag = request.selected_animal_id

        if not extracted_tag:
            tag_match = TAG_ID_PATTERN.search(clean_message)
            if tag_match:
                extracted_tag = tag_match.group(1)

        selected_cattle = None
        milk_history_records = []
        vaccination_recommendations = []
        lactation_status = None
        latest_analysis_records = []
        is_tag_query = extracted_tag is not None

        if extracted_tag and user_id:
            norm_tag = normalize_tag_id(extracted_tag)
            # Query repository for cattle owned strictly by this authenticated user
            selected_cattle = repo.get_cattle(
                animal_id_or_tag=norm_tag,
                user_id=user_id,
                farm_id=request.farm_id
            )

            if selected_cattle:
                # Retrieve persistent milk history for this cow
                milk_history_records = repo.list_milk_records(norm_tag, user_id=user_id, limit=14)
                # Retrieve vaccination schedule and recommendations
                admin_vaccines = repo.list_vaccination_records(norm_tag, user_id=user_id)
                vaccination_recommendations = vaccination_service.generate_recommendations(selected_cattle, admin_vaccines)
                # Retrieve calculated lactation timeline
                lactation_status = lactation_service.calculate_lactation_status(selected_cattle)
                # Retrieve latest analysis records
                latest_analysis_records = repo.get_latest_analysis(
                    user_id=user_id,
                    animal_id=norm_tag,
                    limit=5
                )
            else:
                # Tag ID explicitly queried but not found in user's authorized herd
                # Return explicit clear not-found response (do NOT hallucinate records)
                not_found_replies = {
                    "en": f"I couldn't find a cattle record for Tag ID {extracted_tag}.",
                    "ta": f"Tag ID {extracted_tag} கொண்ட மாட்டின் விவரங்கள் எதுவும் உங்கள் பண்ணைப் பதிவேட்டில் கிடைக்கவில்லை.",
                    "hi": f"मुझे टैग आईडी {extracted_tag} के लिए कोई पशु रिकॉर्ड नहीं मिला।",
                    "te": f"ట్యాగ్ ఐడి {extracted_tag} కోసం పశువు రికార్డు కనుగొనబడలేదు.",
                    "kn": f"ಟ್ಯಾಗ್ ಐಡಿ {extracted_tag} ಗಾಗಿ ಯಾವುದೇ ಜಾನುವಾರು ದಾಖಲೆ ಕಂಡುಬಂದಿಲ್ಲ.",
                    "ml": f"ടാഗ് ഐഡി {extracted_tag} ഉള്ള പശുവിന്റെ രേഖകൾ കണ്ടെത്താനായില്ല.",
                    "mr": f"टॅग आयडी {extracted_tag} साठी कोणतीही नोंद आढळली नाही.",
                    "bn": f"ট্যাগ আইডি {extracted_tag} এর জন্য কোনো পশুর রেকর্ড পাওয়া যায়নি।",
                    "gu": f"ટેગ આઈડી {extracted_tag} માટે કોઈ પશુ રેકોર્ડ મળ્યો નથી.",
                    "pa": f"ਟੈਗ ਆਈਡੀ {extracted_tag} ਲਈ ਕੋਈ ਪਸ਼ੂ ਰਿਕਾਰਡ ਨਹੀਂ ਮਿਲਿਆ।"
                }
                not_found_msg = not_found_replies.get(target_lang, f"I couldn't find a cattle record for Tag ID {extracted_tag}.")

                session_service.record_interaction(
                    session_id=session.id,
                    user_message=clean_message,
                    assistant_reply=not_found_msg,
                    language=target_lang,
                    intent="cattle_tag_lookup",
                    module="cattle"
                )

                return ChatResponse(
                    success=True,
                    reply=not_found_msg,
                    language=target_lang,
                    detected_language=detected_lang,
                    intent="cattle_tag_lookup",
                    module="cattle",
                    session_id=session.id,
                    metadata={
                        "tag_id_queried": extracted_tag,
                        "tag_id_found": False,
                        "provider": "domain_cattle_lookup"
                    }
                )

        # 6. Intent Classification
        intent_match = intent_service.classify(clean_message, conversation_history=history_dicts)
        intent = intent_match.intent
        module = intent_match.module

        # If a valid cattle Tag ID was found and the user is asking about it, adjust module/intent
        if selected_cattle and intent in ["unknown", "greeting", "animal_profile"]:
            intent = "cattle_profile_query"
            module = "cattle"

        # 7. Module Routing & Evaluation
        silage_data = None
        nutrition_data = None
        extracted_metadata: Dict[str, Any] = {
            "intent_confidence": intent_match.confidence,
            "matched_keywords": intent_match.matched_keywords,
            "selected_animal_id": extracted_tag,
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

        # 8. AI Response Generation (passing rich grounded cattle context)
        reply = ai_service.generate_response(
            user_message=clean_message,
            target_language=target_lang,
            intent=intent,
            module=module,
            conversation_history=history_dicts,
            silage_data=silage_data,
            nutrition_data=nutrition_data,
            selected_cattle=selected_cattle,
            milk_history=milk_history_records,
            vaccination_data=vaccination_recommendations,
            lactation_data=lactation_status,
            analysis_records=latest_analysis_records
        )

        extracted_metadata["provider"] = ai_service.last_provider
        if ai_service.last_model_used:
            extracted_metadata["ai_model"] = ai_service.last_model_used
        if ai_service.last_error:
            extracted_metadata["fallback_reason"] = ai_service.last_error

        # 9. Record interaction in session history
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
