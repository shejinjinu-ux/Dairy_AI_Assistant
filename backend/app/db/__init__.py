"""
Database and Persistence Package for Dairy AI Assistant
"""

from backend.app.db.chat_repository import (
    ChatRepository,
    InMemoryChatRepository,
    SupabaseChatRepository,
    ChatPersistenceError,
    SupabaseConnectionError,
    SupabaseQueryError,
    SessionAccessDeniedError,
    get_chat_repository,
    set_chat_repository,
    reset_chat_repository,
)

__all__ = [
    "ChatRepository",
    "InMemoryChatRepository",
    "SupabaseChatRepository",
    "ChatPersistenceError",
    "SupabaseConnectionError",
    "SupabaseQueryError",
    "SessionAccessDeniedError",
    "get_chat_repository",
    "set_chat_repository",
    "reset_chat_repository",
]
