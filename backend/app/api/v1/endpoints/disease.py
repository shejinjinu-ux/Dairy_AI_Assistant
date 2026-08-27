"""
Cattle Disease Diagnosis Endpoint
"""

from fastapi import APIRouter, File, UploadFile, status
from backend.app.schemas.disease import DiseasePredictionResponse
from backend.app.services.disease_service import disease_service
from backend.app.core.exceptions import ImageProcessingError

router = APIRouter(prefix="/predict/disease", tags=["Veterinary Disease Diagnosis"])


@router.post(
    "",
    response_model=DiseasePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Diagnose Cattle Disease from Image"
)
async def predict_disease(
    file: UploadFile = File(..., description="Cattle lesion or clinical image file (JPEG, PNG, WebP)")
):
    """
    Accepts an uploaded bovine clinical image and runs inference through the
    production EfficientNet-B3 model (accuracy: 98.93%) to classify condition as:
    - **FMD**: Foot-and-Mouth Disease
    - **IBK**: Infectious Bovine Keratoconjunctivitis (Pinkeye)
    - **LSD**: Lumpy Skin Disease
    - **Normal**: Healthy animal
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ImageProcessingError(f"Uploaded file '{file.filename}' is not a valid image format.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise ImageProcessingError("Uploaded image file is empty.")

    return disease_service.predict(image_bytes)
