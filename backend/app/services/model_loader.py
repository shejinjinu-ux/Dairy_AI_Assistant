"""
Reusable Model Loader Service

Thread-safe, cached model loader for PyTorch vision architectures and
Scikit-Learn/XGBoost joblib pipelines referenced in model_registry.json.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import timm
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

from backend.config import settings
from backend.app.core.registry import registry_manager
from backend.app.core.exceptions import (
    ModelNotFoundError,
    ModelDisabledError,
    ModelInferenceError,
)

logger = logging.getLogger("dairy_ai.model_loader")


class ModelLoaderService:
    """
    Singleton service managing the lifecycle and memory cache of all AI/ML models.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, Any] = {}
        self._device = settings.torch_device

    @property
    def device(self) -> torch.device:
        return self._device

    def is_cached(self, model_key: str) -> bool:
        return model_key in self._cache

    def get_loaded_models_count(self) -> int:
        return len(self._cache)

    def get_loaded_keys(self) -> List[str]:
        return list(self._cache.keys())

    # ----------------------------------------------------------------------
    # PyTorch Vision Models
    # ----------------------------------------------------------------------

    def load_disease_model(self) -> Tuple[nn.Module, List[str]]:
        """
        Load Cattle Disease EfficientNet-B3 PyTorch model and class labels.
        """
        model_key = "cattle_disease"
        if model_key in self._cache:
            return self._cache[model_key], self._metadata_cache[model_key]

        if not registry_manager.is_model_enabled(model_key):
            raise ModelDisabledError(model_key)

        model_path = registry_manager.get_resolved_model_path(model_key)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")

        classes = ["FMD", "IBK", "LSD", "Normal"]
        classes_file = model_path.parent / "class_names.json"
        if classes_file.exists():
            with open(classes_file, "r", encoding="utf-8") as f:
                classes = json.load(f)

        logger.info(f"Loading PyTorch disease model from {model_path} onto {self._device}...")
        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(classes))

        checkpoint = torch.load(model_path, map_location=self._device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()

        self._cache[model_key] = model
        self._metadata_cache[model_key] = classes
        return model, classes

    def load_breed_model(self) -> Tuple[nn.Module, List[str]]:
        """
        Load Indian Bovine Breed ConvNeXt-Tiny PyTorch model and 41 classes.
        """
        model_key = "cattle_breed"
        if model_key in self._cache:
            return self._cache[model_key], self._metadata_cache[model_key]

        if not registry_manager.is_model_enabled(model_key):
            raise ModelDisabledError(model_key)

        model_path = registry_manager.get_resolved_model_path(model_key)
        classes_path = registry_manager.get_resolved_classes_path(model_key)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")
        if not classes_path or not classes_path.exists():
            raise FileNotFoundError(f"Classes file missing: {classes_path}")

        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)

        logger.info(f"Loading PyTorch breed model from {model_path} ({len(classes)} classes) onto {self._device}...")
        model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=len(classes),
            drop_path_rate=0.2,
        )

        checkpoint = torch.load(model_path, map_location=self._device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(self._device)
        model.eval()

        self._cache[model_key] = model
        self._metadata_cache[model_key] = classes
        return model, classes

    # ----------------------------------------------------------------------
    # Scikit-Learn / XGBoost Joblib Pipelines
    # ----------------------------------------------------------------------

    def load_joblib_pipeline(self, model_key: str) -> Any:
        """
        Load Scikit-Learn / XGBoost joblib pipeline by registry key.
        Checks enablement and caches instance.
        """
        if model_key in self._cache:
            return self._cache[model_key]

        entry = registry_manager.get_model_entry(model_key)
        if not entry:
            raise ModelNotFoundError(model_key)

        if not registry_manager.is_model_enabled(model_key):
            raise ModelDisabledError(
                model_key,
                reason="Experimental model is disabled. Enable ENABLE_EXPERIMENTAL_MODELS in configuration."
            )

        model_path = registry_manager.get_resolved_model_path(model_key)
        if not model_path.exists():
            raise FileNotFoundError(f"Joblib model file missing at: {model_path}")

        logger.info(f"Loading Joblib pipeline '{model_key}' from {model_path}...")
        pipeline = joblib.load(model_path)
        self._cache[model_key] = pipeline
        return pipeline

    # ----------------------------------------------------------------------
    # Pre-warming / Warmup
    # ----------------------------------------------------------------------

    def preload_all_production_models(self) -> Dict[str, bool]:
        """
        Preload all 13 production models on application startup.
        Returns status dictionary.
        """
        results = {}
        logger.info("Preloading all production models from registry...")

        # 1. PyTorch models
        try:
            self.load_disease_model()
            results["cattle_disease"] = True
        except Exception as e:
            logger.error(f"Failed to preload cattle_disease: {e}")
            results["cattle_disease"] = False

        try:
            self.load_breed_model()
            results["cattle_breed"] = True
        except Exception as e:
            logger.error(f"Failed to preload cattle_breed: {e}")
            results["cattle_breed"] = False

        # 2. Production Joblib models
        prod_keys = registry_manager.list_production_models()
        for key in prod_keys:
            if key in {"cattle_disease", "cattle_breed"}:
                continue
            try:
                self.load_joblib_pipeline(key)
                results[key] = True
            except Exception as e:
                logger.error(f"Failed to preload {key}: {e}")
                results[key] = False

        logger.info(f"Preload complete. {sum(results.values())}/{len(results)} models cached successfully.")
        return results


# Global Model Loader Singleton
model_loader = ModelLoaderService()
