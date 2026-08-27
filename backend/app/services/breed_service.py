"""
Indian Bovine Breed Classification Inference Service
"""

from typing import Optional
import io
import torch
from PIL import Image
from torchvision import transforms
from backend.config import settings
from backend.app.services.model_loader import model_loader
from backend.app.schemas.breed import BreedPredictionResponse, TopBreedPrediction
from backend.app.core.exceptions import ImageProcessingError, ModelInferenceError


class BreedInferenceService:
    """Service handling Indian bovine breed classification."""

    def __init__(self):
        self.image_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(
        self,
        image_bytes: bytes,
        threshold: Optional[float] = None
    ) -> BreedPredictionResponse:
        """
        Classify cattle/buffalo breed from image bytes with confidence thresholding.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ImageProcessingError(f"Failed to decode image: {str(e)}")

        model, classes = model_loader.load_breed_model()
        device = model_loader.device
        conf_threshold = threshold if threshold is not None else settings.BREED_CONFIDENCE_THRESHOLD

        try:
            tensor = self.transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, index = probabilities.max(dim=1)
                top_values, top_indices = probabilities.topk(min(5, len(classes)), dim=1)
            del tensor
            del logits

            top_predictions = []
            for value, idx in zip(top_values[0], top_indices[0]):
                prob_val = float(value.item())
                top_predictions.append(
                    TopBreedPrediction(
                        breed=classes[idx.item()],
                        confidence=round(prob_val, 4),
                        confidence_percentage=round(prob_val * 100.0, 2)
                    )
                )

            top_breed = classes[index.item()]
            top_conf = float(confidence.item())

            is_confident = top_conf >= conf_threshold
            breed_status = "identified" if is_confident else "uncertain"
            predicted_breed = top_breed if is_confident else None
            recommendation = (
                None
                if is_confident
                else "Upload a clearer side-profile image with the full animal visible."
            )

            return BreedPredictionResponse(
                breed_status=breed_status,
                predicted_breed=predicted_breed,
                confidence=round(top_conf, 4),
                confidence_percentage=round(top_conf * 100.0, 2),
                recommendation=recommendation,
                top_5_predictions=top_predictions,
                total_classes_supported=len(classes),
                model_architecture="convnext_tiny",
                device_used=str(device)
            )

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ModelInferenceError("cattle_breed", str(e))


breed_service = BreedInferenceService()
