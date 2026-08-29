"""
Multilingual AI Chatbot Endpoints
"""

import logging
from fastapi import APIRouter, Request, HTTPException, status
from backend.app.schemas.chat import ChatRequest, ChatResponse, ChatErrorResponse
from backend.app.services.chat.chat_service import chat_service
from backend.app.db.chat_repository import SessionAccessDeniedError, ChatPersistenceError
from backend.app.core.ownership_guard import ownership_guard
from backend.app.core.exceptions import AppBaseException

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
async def process_chat(request: Request, payload: ChatRequest):
    """
    Communicates with the Multilingual Dairy AI Assistant in 20+ Indian languages.

    - **message**: Farmer message in native script or Romanized Tanglish/Hinglish (required).
    - **language**: Optional explicit ISO language code (e.g. `ta`, `hi`, `te`, `kn`, `ml`, `en`). If omitted, detected automatically.
    - **session_id**: Optional session ID to maintain conversation memory across turns.
    - **user_id**: Optional authenticated user ID.
    - **farm_id**: Optional farm ID.
    - **selected_animal_id**: Optional selected animal ID to restrict advisory to that specific authorized animal.
    """
    try:
        # Derive authenticated user identity and validate ownership if farm_id or selected_animal_id provided
        auth_user_id = ownership_guard.extract_authenticated_user_id(request) or payload.user_id
        if auth_user_id:
            payload.user_id = auth_user_id
            
        ownership_guard.validate_request_ownership(
            request=request,
            farm_id=payload.farm_id,
            animal_id=payload.selected_animal_id,
            require_auth=False
        )
        response = chat_service.process_message(payload, http_request=request)
        return response
    except HTTPException:
        raise
    except AppBaseException:
        raise
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
