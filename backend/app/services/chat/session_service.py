"""
Conversation Session & Context Memory Service
Manages chat session lifecycle, conversational context window, and user session ownership.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.app.db.chat_repository import (
    ChatRepository,
    SessionAccessDeniedError,
    get_chat_repository,
)
from backend.app.schemas.chat import ChatMessageSchema, ChatSessionSchema

logger = logging.getLogger("dairy_ai.chat.session_service")


class SessionService:
    """Manages chat session lifecycle and conversational context window."""

    @property
    def repository(self) -> ChatRepository:
        """Dynamically resolve the active ChatRepository instance."""
        return get_chat_repository()

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        language: str = "en"
    ) -> ChatSessionSchema:
        """
        Retrieves existing session or creates a new session.
        Enforces user ownership security when user_id is provided.
        """
        if session_id:
            existing = self.repository.get_session(session_id)
            if existing:
                # Security Check: Prevent unauthorized access to another user's session
                if existing.user_id and user_id and existing.user_id != user_id:
                    logger.warning(
                        f"Unauthorized access attempt: user '{user_id}' requested session '{session_id}' owned by '{existing.user_id}'."
                    )
                    raise SessionAccessDeniedError(
                        f"Access denied: Session '{session_id}' belongs to another user."
                    )

                if user_id and not existing.user_id:
                    existing.user_id = user_id

                existing.language = language
                self.repository.save_session(existing)
                return existing

        # Generate new session ID if not supplied or not existing
        new_id = session_id or f"sess_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)
        new_session = ChatSessionSchema(
            id=new_id,
            user_id=user_id,
            language=language,
            created_at=now,
            updated_at=now,
            messages=[]
        )
        return self.repository.save_session(new_session)

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessageSchema]:
        """Fetches recent conversation history in chronological order."""
        max_history = limit or settings.CHAT_MAX_HISTORY_MESSAGES
        return self.repository.get_recent_messages(session_id, limit=max_history)

    def record_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        language: str,
        intent: str,
        module: str
    ) -> None:
        """Saves user message and assistant reply to the session history."""
        now = datetime.now(timezone.utc)

        # 1. Record User Message
        user_created_at = datetime.now(timezone.utc)
        user_msg = ChatMessageSchema(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            role="user",
            message=user_message,
            language=language,
            intent=intent,
            module=module,
            created_at=user_created_at
        )
        self.repository.add_message(session_id, user_msg)

        # 2. Record Assistant Message
        assistant_created_at = datetime.now(timezone.utc)
        assistant_msg = ChatMessageSchema(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            role="assistant",
            message=assistant_reply,
            language=language,
            intent=intent,
            module=module,
            created_at=assistant_created_at
        )
        self.repository.add_message(session_id, assistant_msg)


session_service = SessionService()
