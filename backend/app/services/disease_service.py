"""
Cattle Disease Diagnostic Inference Service
"""

import io
import torch
from PIL import Image
from torchvision import transforms
from backend.app.services.model_loader import model_loader
from backend.app.schemas.disease import DiseasePredictionResponse
from backend.app.core.exceptions import ImageProcessingError, ModelInferenceError


DISEASE_NAME_MAP = {
    "FMD": "Foot-and-Mouth Disease (Aphthovirus)",
    "IBK": "Infectious Bovine Keratoconjunctivitis (Pinkeye)",
    "LSD": "Lumpy Skin Disease (Capripoxvirus)",
    "Normal": "Normal / Healthy (No Detected Pathology)"
}


class DiseaseInferenceService:
    """Service handling cattle disease vision diagnosis."""

    def __init__(self):
        self.image_size = 300
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_bytes: bytes) -> DiseasePredictionResponse:
        """
        Diagnose cattle disease from image bytes.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ImageProcessingError(f"Failed to decode image: {str(e)}")

        model, classes = model_loader.load_disease_model()
        device = model_loader.device

        try:
            tensor = self.transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

            max_idx = max(range(len(classes)), key=probabilities.__getitem__)
            predicted_class = classes[max_idx]
            confidence = float(probabilities[max_idx])
            prob_dict = {cls_name: float(prob) for cls_name, prob in zip(classes, probabilities)}

            is_disease = predicted_class != "Normal"
            full_name = DISEASE_NAME_MAP.get(predicted_class, predicted_class)

            return DiseasePredictionResponse(
                predicted_class=predicted_class,
                confidence=round(confidence, 4),
                confidence_percentage=round(confidence * 100.0, 2),
                is_disease_detected=is_disease,
                disease_name_full=full_name,
                probabilities=prob_dict,
                model_version="efficientnet_b3",
                device_used=str(device)
            )

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ModelInferenceError("cattle_disease", str(e))


disease_service = DiseaseInferenceService()
