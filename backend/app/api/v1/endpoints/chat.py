"""
Multilingual AI Chatbot Endpoints
"""

import logging
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.chat import ChatRequest, ChatResponse, ChatErrorResponse
from backend.app.services.chat.chat_service import chat_service
from backend.app.db.chat_repository import SessionAccessDeniedError, ChatPersistenceError

logger = logging.getLogger("dairy_ai.api.chat")

router = APIRouter(prefix="/chat", tags=["Multilingual AI Chat Assistant"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Multilingual Dairy AI Farmer Chat",
    responses={
        200: {"model": ChatResponse, "description": "Successful chat interaction response."},
        403: {"model": ChatErrorResponse, "description": "Unauthorized access to session."},
        422: {"model": ChatErrorResponse, "description": "Validation error or invalid message."},
        500: {"model": ChatErrorResponse, "description": "Internal server or persistence error."}
    }
)
async def process_chat(payload: ChatRequest):
    """
    Communicates with the Multilingual Dairy AI Assistant in 20+ Indian languages.

    - **message**: Farmer message in native script or Romanized Tanglish/Hinglish (required).
    - **language**: Optional explicit ISO language code (e.g. `ta`, `hi`, `te`, `kn`, `ml`, `en`). If omitted, detected automatically.
    - **session_id**: Optional session ID to maintain conversation memory across turns.
    - **user_id**: Optional authenticated user ID (e.g. from Supabase).

    **Supported Capabilities**:
    1. Automatic language detection and localized responses across 20+ Indian languages.
    2. Intelligent intent classification across 9 dairy domains.
    3. Seamless routing to Silage quality evaluation.
    4. Pluggable Field Nutrition ration formulation interface.
    5. Multi-turn conversational memory with Supabase persistence.
    """
    try:
        response = chat_service.process_message(payload)
        return response
    except SessionAccessDeniedError as e:
        logger.warning(f"Session access denied: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to access this session."
        )
    except ChatPersistenceError as e:
        logger.error(f"Persistence error during chat processing: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A persistent storage error occurred while processing the conversation. Please try again."
        )
    except Exception as e:
        logger.error(f"Unexpected error processing chat message: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating your chat response. Please try again."
        )
