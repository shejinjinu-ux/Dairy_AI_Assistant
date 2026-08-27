"""
Model Registry & Status Inspection Endpoints
"""

from fastapi import APIRouter
from backend.config import settings
from backend.app.schemas.models_status import ModelRegistryStatusResponse, ModelDescriptor
from backend.app.core.registry import registry_manager
from backend.app.services.model_loader import model_loader
from backend.app.core.exceptions import ModelNotFoundError

router = APIRouter(prefix="/models", tags=["Model Registry"])


def build_descriptor(key: str, info: dict) -> ModelDescriptor:
    """Helper to convert raw registry item into ModelDescriptor schema."""
    return ModelDescriptor(
        key=key,
        status=info.get("status", "unknown"),
        framework=info.get("framework", "unknown"),
        task=info.get("task", "unknown"),
        path=info.get("path", ""),
        is_enabled=registry_manager.is_model_enabled(key),
        is_cached_in_memory=model_loader.is_cached(key),
        target=info.get("target"),
        outputs=info.get("outputs"),
        classes_count=info.get("classes"),
        accuracy=info.get("accuracy"),
        r2_score=info.get("r2"),
        f1_macro=info.get("f1_macro")
    )


@router.get("", response_model=ModelRegistryStatusResponse, summary="List All Registered Models")
async def list_models():
    """
    List all 15 models registered in model_registry.json along with their
    status (production/experimental), framework, enablement flag, and memory caching status.
    """
    models_dict = registry_manager.all_models
    descriptors = [build_descriptor(k, v) for k, v in models_dict.items()]

    prod_count = len(registry_manager.list_production_models())
    exp_count = len(registry_manager.list_experimental_models())

    return ModelRegistryStatusResponse(
        version=registry_manager.version,
        total_models=len(descriptors),
        production_count=prod_count,
        experimental_count=exp_count,
        experimental_enabled=settings.ENABLE_EXPERIMENTAL_MODELS,
        models=descriptors
    )


@router.get("/{model_id}", response_model=ModelDescriptor, summary="Get Model Details by Key")
async def get_model(model_id: str):
    """
    Retrieve full registry and runtime metadata for a specific model key.
    """
    entry = registry_manager.get_model_entry(model_id)
    if not entry:
        raise ModelNotFoundError(model_id)

    return build_descriptor(model_id, entry)
