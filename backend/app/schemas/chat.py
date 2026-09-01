"""
Multilingual AI Chatbox Schemas
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ChatRequest(BaseModel):
    """Input payload for multilingual chat interaction."""
    message: str = Field(
        ...,
        description="The message from the farmer in any supported script or Romanized text.",
        min_length=1,
        max_length=2000,
        examples=["En maadu 15 litre paal kudukuthu, enna feed kudukkanum?", "Tell me about COW-1001", "What should I feed my cow?"]
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional ISO 639-1 / standard language code (e.g. 'ta', 'hi', 'te', 'en'). If omitted, language is detected automatically.",
        examples=["ta", "hi", "en"]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier to maintain multi-turn conversation memory.",
        examples=["sess_123456789"]
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional authenticated user identifier (e.g. from Supabase auth).",
        examples=["usr_987654321"]
    )
    farm_id: Optional[str] = Field(
        default=None,
        description="Optional farm identifier to scope context.",
        examples=["farm_123"]
    )
    selected_animal_id: Optional[str] = Field(
        default=None,
        description="Optional selected animal identifier / Tag ID (e.g. 'COW-1001').",
        examples=["COW-1001"]
    )
    tag_id: Optional[str] = Field(
        default=None,
        description="Optional alias for selected cattle Tag ID.",
        examples=["COW-1001"]
    )

    @model_validator(mode="before")
    @classmethod
    def sync_tag_id_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            tid = data.get("tag_id") or data.get("selected_animal_id")
            if tid:
                cleaned = str(tid).strip()
                data["selected_animal_id"] = cleaned
                data["tag_id"] = cleaned
        return data

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or whitespace only.")
        return cleaned


class ChatResponse(BaseModel):
    """Consistent output schema for chat responses."""
    success: bool = Field(default=True, description="Indicates whether the query was processed successfully.")
    reply: str = Field(..., description="The localized, farmer-friendly AI response.")
    language: str = Field(..., description="The language code used for the reply.")
    detected_language: str = Field(..., description="The automatically detected input language code.")
    intent: str = Field(..., description="Classified intent (e.g., 'nutrition', 'silage_quality', 'greeting', etc.).")
    module: str = Field(..., description="Target service/module routed to (e.g., 'nutrition', 'silage', 'chat', etc.).")
    session_id: str = Field(..., description="Active or newly generated conversation session ID.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional execution telemetry, extracted entities, or module status."
    )


class ChatErrorDetail(BaseModel):
    """Detailed error object for chat failures."""
    code: str = Field(..., description="Machine-readable error code (e.g., 'INVALID_MESSAGE', 'UNSUPPORTED_LANGUAGE').")
    message: str = Field(..., description="Human-readable explanation of the error.")


class ChatErrorResponse(BaseModel):
    """Uniform error response structure for chat failures."""
    success: bool = Field(default=False, description="Always False for error responses.")
    error: ChatErrorDetail = Field(..., description="Error description details.")


class ChatMessageSchema(BaseModel):
    """Individual message entity in a chat conversation."""
    id: Optional[str] = Field(default=None, description="Unique message ID.")
    session_id: str = Field(..., description="Parent session ID.")
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'.")
    message: str = Field(..., description="Message text content.")
    language: str = Field(default="en", description="Language of the message.")
    intent: Optional[str] = Field(default=None, description="Associated intent if applicable.")
    module: Optional[str] = Field(default=None, description="Associated module if applicable.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the message.")


class ChatSessionSchema(BaseModel):
    """Conversation session containing metadata and recent message history."""
    id: str = Field(..., description="Session identifier.")
    user_id: Optional[str] = Field(default=None, description="Associated user ID if authenticated.")
    language: str = Field(default="en", description="Primary preferred session language.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Session creation time.")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Session last updated time.")
    messages: List[ChatMessageSchema] = Field(default_factory=list, description="Recent conversation messages list.")
