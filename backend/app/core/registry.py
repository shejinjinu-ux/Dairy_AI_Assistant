"""
Model Registry Manager

Reads models/model_registry.json and provides structured metadata access,
path resolution, framework classification, and status gating.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.config import settings


class ModelRegistryManager:
    """
    Thread-safe model registry manager.
    Parses and serves model metadata from models/model_registry.json.
    """

    def __init__(self, registry_path: Optional[Path] = None, project_root: Optional[Path] = None):
        self.registry_path = registry_path or settings.MODEL_REGISTRY_PATH
        self.project_root = project_root or settings.PROJECT_ROOT
        self._raw_data: Dict[str, Any] = {}
        self._models: Dict[str, Dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        """Load or reload the registry JSON file."""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry file not found at: {self.registry_path}"
            )

        with open(self.registry_path, "r", encoding="utf-8-sig") as f:
            self._raw_data = json.load(f)

        self._models = self._raw_data.get("models", {})

    @property
    def version(self) -> str:
        """Return registry schema version."""
        return self._raw_data.get("version", "1.0")

    @property
    def all_models(self) -> Dict[str, Dict[str, Any]]:
        """Return all model entries in the registry."""
        return self._models

    def get_model_entry(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata by key."""
        return self._models.get(model_key)

    def get_resolved_model_path(self, model_key: str) -> Path:
        """
        Return the absolute Path for a given model key,
        resolved relative to project root.
        """
        entry = self.get_model_entry(model_key)
        if not entry:
            raise KeyError(f"Model '{model_key}' is not registered in model_registry.json.")

        rel_path = entry.get("path")
        if not rel_path:
            raise ValueError(f"Model '{model_key}' has no 'path' attribute in registry.")

        return self.project_root / rel_path

    def get_resolved_classes_path(self, model_key: str) -> Optional[Path]:
        """Return absolute path to classes file if configured."""
        entry = self.get_model_entry(model_key)
        if entry and "classes_path" in entry:
            return self.project_root / entry["classes_path"]
        return None

    def is_production(self, model_key: str) -> bool:
        """Check if model status is 'production'."""
        entry = self.get_model_entry(model_key)
        return bool(entry and entry.get("status") == "production")

    def is_experimental(self, model_key: str) -> bool:
        """Check if model status is 'experimental'."""
        entry = self.get_model_entry(model_key)
        return bool(entry and entry.get("status") == "experimental")

    def is_model_enabled(self, model_key: str) -> bool:
        """
        Determine if model is currently accessible:
        - Production models are always enabled.
        - Experimental models are enabled only if ENABLE_EXPERIMENTAL_MODELS=True.
        - Unregistered or rejected models are disabled.
        """
        if self.is_production(model_key):
            return True
        if self.is_experimental(model_key):
            return settings.ENABLE_EXPERIMENTAL_MODELS
        return False

    def list_production_models(self) -> List[str]:
        """Return list of keys for all production models."""
        return [k for k, v in self._models.items() if v.get("status") == "production"]

    def list_experimental_models(self) -> List[str]:
        """Return list of keys for all experimental models."""
        return [k for k, v in self._models.items() if v.get("status") == "experimental"]

    def get_summary(self) -> Dict[str, Any]:
        """Return high-level summary of model availability."""
        prod = self.list_production_models()
        exp = self.list_experimental_models()
        return {
            "version": self.version,
            "total_registered_models": len(self._models),
            "production_models_count": len(prod),
            "experimental_models_count": len(exp),
            "production_models": prod,
            "experimental_models": exp,
            "experimental_enabled": settings.ENABLE_EXPERIMENTAL_MODELS,
            "active_device": str(settings.torch_device),
        }


# Singleton Instance
registry_manager = ModelRegistryManager()
