"""
Backend Application Configuration
"""

from pathlib import Path
from typing import Any, List
import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project Root Directory Resolution
# config.py is at <project_root>/backend/config.py -> parents[1] is project root
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """
    Application Settings for Dairy AI Assistant Backend
    """
    model_config = SettingsConfigDict(
        env_file=(DEFAULT_PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Metadata
    PROJECT_NAME: str = "Dairy AI Assistant API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = (
        "Production-grade Dairy AI Assistant API providing high-precision veterinary "
        "disease diagnosis, Indian bovine breed classification, lactation milk yield estimation, "
        "silage quality & fermentation indexing, multi-target feed nutrition analysis, "
        "and NIR milk composition spectroscopy."
    )

    # Path Configuration (Relative to Project Root)
    PROJECT_ROOT: Path = DEFAULT_PROJECT_ROOT
    MODEL_REGISTRY_PATH: Path = DEFAULT_PROJECT_ROOT / "models" / "model_registry.json"

    # Execution Device (CUDA if available and configured, otherwise CPU)
    FORCE_CPU: bool = False

    # Model Prediction Confidence Thresholds
    BREED_CONFIDENCE_THRESHOLD: float = 0.70

    # Experimental Model Gate
    # When False, experimental models (e.g. milk_quality_protein, mycotoxin_don) remain disabled
    ENABLE_EXPERIMENTAL_MODELS: bool = False

    # Security & CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "https://dairy-nova-ai-gilt.vercel.app",
        "https://dairy-nova-ai.vercel.app",
        "capacitor://localhost",
        "*"
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, tuple)):
            return list(v)
        return [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8000",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8000",
            "https://dairy-nova-ai-gilt.vercel.app",
            "https://dairy-nova-ai.vercel.app",
            "capacitor://localhost",
            "*"
        ]

    # Multilingual Chat & AI Configuration
    CHAT_DEFAULT_LANGUAGE: str = "en"
    CHAT_MAX_MESSAGE_LENGTH: int = 2000
    CHAT_MAX_HISTORY_MESSAGES: int = 10
    AI_PROVIDER: str = "local"  # "local", "gemini", "openai"
    AI_API_KEY: str | None = None
    AI_MODEL: str | None = None
    TRANSLATION_API_KEY: str | None = None

    # Supabase / Database Configuration (Persistent storage)
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    SUPABASE_SERVICE_ROLE_KEY: str | None = None
    SUPABASE_SERVICE_KEY: str | None = None
    SUPABASE_SECRET_KEY: str | None = None
    SUPABASE_ANON_KEY: str | None = None
    SUPABASE_PUBLISHABLE_KEY: str | None = None
    SUPABASE_PUBLIC_KEY: str | None = None

    @property
    def effective_supabase_url(self) -> str | None:
        """Resolve clean Supabase base URL."""
        if self.SUPABASE_URL and str(self.SUPABASE_URL).strip():
            return str(self.SUPABASE_URL).strip()
        return None

    @property
    def effective_supabase_key(self) -> str | None:
        """Resolve the effective Supabase API key with service role precedence."""
        for candidate in [
            self.SUPABASE_SERVICE_ROLE_KEY,
            self.SUPABASE_SERVICE_KEY,
            self.SUPABASE_SECRET_KEY,
            self.SUPABASE_KEY,
            self.SUPABASE_ANON_KEY,
            self.SUPABASE_PUBLISHABLE_KEY,
            self.SUPABASE_PUBLIC_KEY
        ]:
            if candidate and str(candidate).strip():
                return str(candidate).strip()
        return None

    @property
    def is_supabase_configured(self) -> bool:
        """Returns True if valid Supabase URL and Key are configured."""
        return bool(self.effective_supabase_url and self.effective_supabase_key)

    @property
    def torch_device(self) -> torch.device:
        """Resolve PyTorch inference device."""
        if not self.FORCE_CPU and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


settings = Settings()

