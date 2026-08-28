"""
Chat Session & Message Repository Interface and Implementations
Provides production-grade Supabase PostgreSQL and In-Memory persistence.
"""

import abc
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import httpx

from backend.config import settings
from backend.app.schemas.chat import ChatMessageSchema, ChatSessionSchema

logger = logging.getLogger("dairy_ai.db.chat_repository")


class ChatPersistenceError(Exception):
    """Base exception for chat persistence errors."""
    pass


class SupabaseConnectionError(ChatPersistenceError):
    """Raised when connecting or authenticating to Supabase fails."""
    pass


class SupabaseQueryError(ChatPersistenceError):
    """Raised when a Supabase REST query fails."""
    pass


class SessionAccessDeniedError(ChatPersistenceError):
    """Raised when an unauthorized user attempts to access a protected session."""
    pass


class ChatRepository(abc.ABC):
    """Abstract Base Class for Chat Session & Message Persistence."""

    @abc.abstractmethod
    def get_session(self, session_id: str) -> Optional[ChatSessionSchema]:
        """Retrieve a session by its ID."""
        pass

    @abc.abstractmethod
    def save_session(self, session: ChatSessionSchema) -> ChatSessionSchema:
        """Create or update a chat session."""
        pass

    @abc.abstractmethod
    def add_message(self, session_id: str, message: ChatMessageSchema) -> ChatMessageSchema:
        """Append a message to a chat session."""
        pass

    @abc.abstractmethod
    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[ChatMessageSchema]:
        """Fetch the most recent N messages for a session in chronological order."""
        pass

    @abc.abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and its associated messages."""
        pass

    @abc.abstractmethod
    def verify_connection(self) -> bool:
        """Verify database connectivity and table readiness."""
        pass

    def close(self) -> None:
        """Release any open resources or network connection pools."""
        pass


class InMemoryChatRepository(ChatRepository):
    """
    Thread-safe in-memory session and message repository.
    Default repository used for local development, tests, or when Supabase is not configured.
    """

    def __init__(self, max_messages_per_session: int = 50):
        self._sessions: Dict[str, ChatSessionSchema] = {}
        self._lock = threading.RLock()
        self._max_messages = max_messages_per_session

    def get_session(self, session_id: str) -> Optional[ChatSessionSchema]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.model_copy(deep=True)
            return None

    def save_session(self, session: ChatSessionSchema) -> ChatSessionSchema:
        with self._lock:
            session.updated_at = datetime.now(timezone.utc)
            self._sessions[session.id] = session.model_copy(deep=True)
            return self._sessions[session.id]

    def add_message(self, session_id: str, message: ChatMessageSchema) -> ChatMessageSchema:
        with self._lock:
            if not message.id:
                message.id = f"msg_{uuid.uuid4().hex[:12]}"
            if not message.created_at:
                message.created_at = datetime.now(timezone.utc)

            session = self._sessions.get(session_id)
            if not session:
                session = ChatSessionSchema(
                    id=session_id,
                    language=message.language or settings.CHAT_DEFAULT_LANGUAGE,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    messages=[]
                )
                self._sessions[session_id] = session

            session.messages.append(message.model_copy(deep=True))
            if len(session.messages) > self._max_messages:
                session.messages = session.messages[-self._max_messages:]
            session.updated_at = datetime.now(timezone.utc)
            return message

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[ChatMessageSchema]:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.messages:
                return []
            return [m.model_copy(deep=True) for m in session.messages[-limit:]]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def verify_connection(self) -> bool:
        return True


class SupabaseChatRepository(ChatRepository):
    """
    Production Supabase REST repository for persisting chat sessions and messages.
    Uses connection-pooled HTTP client and adheres to PostgREST specification.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        client: Optional[httpx.Client] = None,
        timeout: float = 20.0
    ):
        if not supabase_url or not supabase_key:
            raise ValueError("Both supabase_url and supabase_key are required for SupabaseChatRepository.")

        parsed = urlparse(supabase_url)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or parsed.path.split("/")[0]
        self.base_url = f"{scheme}://{netloc}".rstrip("/")
        self.supabase_url = self.base_url
        self.rest_url = f"{self.base_url}/rest/v1"

        self.supabase_key = supabase_key.strip()
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        self._owned_client = client is None
        self.client = client or httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
        )

    def verify_connection(self) -> bool:
        """
        Validates connectivity, authentication, and table schema readiness.
        """
        try:
            res = self.client.get(
                f"{self.rest_url}/chat_sessions?limit=1",
                headers=self.headers
            )
            if res.status_code == 200:
                logger.info("Supabase connection verified successfully.")
                return True
            elif res.status_code in [401, 403]:
                logger.error(f"Supabase authentication failed with HTTP {res.status_code}.")
                raise SupabaseConnectionError(f"Supabase authentication failed (HTTP {res.status_code}). Check SUPABASE_KEY.")
            elif res.status_code == 404:
                logger.error("Supabase table 'chat_sessions' not found. Please apply supabase_chat_schema.sql.")
                raise SupabaseQueryError("Table 'chat_sessions' does not exist in Supabase.")
            else:
                logger.error(f"Supabase health check returned HTTP {res.status_code}.")
                raise SupabaseConnectionError(f"Supabase connection test failed with status {res.status_code}.")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"Supabase network connection error: {type(e).__name__}")
            raise SupabaseConnectionError(f"Failed to connect to Supabase: {e}")

    def get_session(self, session_id: str) -> Optional[ChatSessionSchema]:
        try:
            res = self.client.get(
                f"{self.rest_url}/chat_sessions?id=eq.{session_id}&select=*,chat_messages(*)",
                headers=self.headers
            )
            if res.status_code == 200:
                data = res.json()
                if not data:
                    return None
                session_data = data[0]
                raw_messages = session_data.get("chat_messages", [])
                raw_messages.sort(key=lambda m: m.get("created_at") or "")
                messages = [ChatMessageSchema(**m) for m in raw_messages]
                return ChatSessionSchema(
                    id=session_data["id"],
                    user_id=session_data.get("user_id"),
                    language=session_data.get("language", "en"),
                    created_at=session_data.get("created_at") or datetime.now(timezone.utc),
                    updated_at=session_data.get("updated_at") or datetime.now(timezone.utc),
                    messages=messages
                )
            elif res.status_code in [401, 403]:
                raise SupabaseConnectionError(f"Supabase authentication error (HTTP {res.status_code}).")
            else:
                raise SupabaseQueryError(f"Failed to get session from Supabase (HTTP {res.status_code}): {res.text}")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"Network error in get_session: {type(e).__name__}")
            raise SupabaseConnectionError(f"Network error contacting Supabase: {e}")

    def save_session(self, session: ChatSessionSchema) -> ChatSessionSchema:
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            created_iso = (
                session.created_at.isoformat()
                if isinstance(session.created_at, datetime)
                else str(session.created_at)
            )
            payload = {
                "id": session.id,
                "user_id": session.user_id,
                "language": session.language,
                "created_at": created_iso,
                "updated_at": now_iso
            }
            res = self.client.post(
                f"{self.rest_url}/chat_sessions?on_conflict=id",
                headers={
                    **self.headers,
                    "Prefer": "resolution=merge-duplicates,return=representation"
                },
                json=payload
            )
            if res.status_code in [200, 201]:
                return session
            elif res.status_code in [401, 403]:
                # If RLS rejects saving user_id when using an anon/publishable key without service_role:
                if payload.get("user_id") is not None:
                    logger.warning(
                        "Supabase RLS blocked saving session with user_id using the configured key. "
                        "Retrying as anonymous session. To persist user_id, configure SUPABASE_SERVICE_ROLE_KEY."
                    )
                    retry_payload = {**payload, "user_id": None}
                    retry_res = self.client.post(
                        f"{self.rest_url}/chat_sessions?on_conflict=id",
                        headers={
                            **self.headers,
                            "Prefer": "resolution=merge-duplicates,return=representation"
                        },
                        json=retry_payload
                    )
                    if retry_res.status_code in [200, 201]:
                        return session
                raise SupabaseConnectionError(f"Supabase auth failed during save_session (HTTP {res.status_code}): {res.text}")
            else:
                raise SupabaseQueryError(f"Failed to save session to Supabase (HTTP {res.status_code}): {res.text}")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"Network error in save_session: {type(e).__name__}")
            raise SupabaseConnectionError(f"Network error contacting Supabase: {e}")

    def add_message(self, session_id: str, message: ChatMessageSchema) -> ChatMessageSchema:
        msg_id = message.id or f"msg_{uuid.uuid4().hex[:12]}"
        message.id = msg_id
        created_at_dt = message.created_at or datetime.now(timezone.utc)
        if isinstance(created_at_dt, datetime):
            created_iso = created_at_dt.isoformat()
        else:
            created_iso = str(created_at_dt)

        try:
            # 1. Ensure parent session exists to satisfy foreign key constraint
            parent_session_payload = {
                "id": session_id,
                "language": message.language or "en",
                "updated_at": created_iso
            }
            res_parent = self.client.post(
                f"{self.rest_url}/chat_sessions?on_conflict=id",
                headers={
                    **self.headers,
                    "Prefer": "resolution=merge-duplicates,return=minimal"
                },
                json=parent_session_payload
            )
            if res_parent.status_code not in [200, 201, 204]:
                logger.debug(f"Parent session upsert returned HTTP {res_parent.status_code}")

            # 2. Insert message into chat_messages
            msg_payload = {
                "id": msg_id,
                "session_id": session_id,
                "role": message.role,
                "message": message.message,
                "language": message.language,
                "intent": message.intent,
                "module": message.module,
                "created_at": created_iso
            }
            res = self.client.post(
                f"{self.rest_url}/chat_messages",
                headers=self.headers,
                json=msg_payload
            )
            if res.status_code in [200, 201]:
                return message
            elif res.status_code in [401, 403]:
                raise SupabaseConnectionError(f"Supabase auth error in add_message (HTTP {res.status_code}): {res.text}")
            else:
                raise SupabaseQueryError(f"Failed to add message in Supabase (HTTP {res.status_code}): {res.text}")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"Network error in add_message: {type(e).__name__}")
            raise SupabaseConnectionError(f"Network error contacting Supabase: {e}")

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[ChatMessageSchema]:
        try:
            res = self.client.get(
                f"{self.rest_url}/chat_messages?session_id=eq.{session_id}&order=created_at.desc&limit={limit}",
                headers=self.headers
            )
            if res.status_code == 200:
                data = res.json()
                data.reverse()
                return [ChatMessageSchema(**m) for m in data]
            elif res.status_code in [401, 403]:
                raise SupabaseConnectionError(f"Supabase auth error in get_recent_messages (HTTP {res.status_code}).")
            else:
                raise SupabaseQueryError(f"Failed to fetch recent messages (HTTP {res.status_code}): {res.text}")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"Network error in get_recent_messages: {type(e).__name__}")
            raise SupabaseConnectionError(f"Network error contacting Supabase: {e}")

    def delete_session(self, session_id: str) -> bool:
        try:
            res = self.client.delete(
                f"{self.rest_url}/chat_sessions?id=eq.{session_id}",
                headers=self.headers
            )
            if res.status_code in [200, 204]:
                return True
            elif res.status_code in [401, 403]:
                raise SupabaseConnectionError(f"Supabase auth error in delete_session (HTTP {res.status_code}).")
            else:
                raise SupabaseQueryError(f"Failed to delete session (HTTP {res.status_code}): {res.text}")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"Network error in delete_session: {type(e).__name__}")
            raise SupabaseConnectionError(f"Network error contacting Supabase: {e}")

    def close(self) -> None:
        if self._owned_client and not self.client.is_closed:
            self.client.close()


# Global singleton repository instance
_chat_repository_instance: Optional[ChatRepository] = None
_repo_lock = threading.Lock()


def get_chat_repository() -> ChatRepository:
    """Factory providing initialized ChatRepository singleton."""
    global _chat_repository_instance
    if _chat_repository_instance is None:
        with _repo_lock:
            if _chat_repository_instance is None:
                if settings.is_supabase_configured:
                    logger.info("Initializing SupabaseChatRepository for persistent storage.")
                    _chat_repository_instance = SupabaseChatRepository(
                        supabase_url=settings.effective_supabase_url,
                        supabase_key=settings.effective_supabase_key
                    )
                else:
                    logger.info("Initializing InMemoryChatRepository for chat sessions.")
                    _chat_repository_instance = InMemoryChatRepository()
    return _chat_repository_instance


def set_chat_repository(repository: ChatRepository) -> None:
    """Explicitly override the active ChatRepository instance (useful for tests/dependency injection)."""
    global _chat_repository_instance
    with _repo_lock:
        _chat_repository_instance = repository


def reset_chat_repository() -> None:
    """Reset repository singleton to re-evaluate configuration."""
    global _chat_repository_instance
    with _repo_lock:
        if _chat_repository_instance:
            _chat_repository_instance.close()
        _chat_repository_instance = None
