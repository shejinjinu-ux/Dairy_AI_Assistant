"""
Indian Bovine Breed Classification Endpoint
"""

from typing import Optional
from fastapi import APIRouter, File, Query, UploadFile, status
from backend.app.schemas.breed import BreedPredictionResponse
from backend.app.services.breed_service import breed_service
from backend.app.core.exceptions import ImageProcessingError

router = APIRouter(prefix="/predict/breed", tags=["Bovine Breed Classification"])


@router.post(
    "",
    response_model=BreedPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Indian Bovine Breed from Image"
)
async def predict_breed(
    file: UploadFile = File(..., description="Bovine body or portrait image (JPEG, PNG, WebP)"),
    confidence_threshold: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence threshold (0.0 to 1.0) overriding server default (0.70)."
    )
):
    """
    Accepts an uploaded image of an Indian cattle or buffalo breed and runs inference
    through the production ConvNeXt-Tiny model across 41 registered Indian bovine breeds.

    **Confidence Thresholding Behavior**:
    - **Top-1 Confidence >= Threshold (default 0.70)**: Returns `breed_status = "identified"` and the predicted breed name.
    - **Top-1 Confidence < Threshold**: Returns `breed_status = "uncertain"`, `predicted_breed = null`, ranked top-5 probabilities, and a recommendation (`"Upload a clearer side-profile image with the full animal visible."`).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ImageProcessingError(f"Uploaded file '{file.filename}' is not a valid image format.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise ImageProcessingError("Uploaded image file is empty.")

    return breed_service.predict(image_bytes, threshold=confidence_threshold)
