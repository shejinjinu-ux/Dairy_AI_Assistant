"""
Cattle Disease Diagnostic Inference Service
"""

import io
import torch
from PIL import Image
from torchvision import transforms
from backend.app.services.model_loader import model_loader
from backend.app.schemas.disease import DiseasePredictionResponse
from backend.app.services.vaccination_service import vaccination_service
from backend.app.core.exceptions import ImageProcessingError, ModelInferenceError


DISEASE_NAME_MAP = {
    "FMD": "Foot-and-Mouth Disease (Aphthovirus)",
    "IBK": "Infectious Bovine Keratoconjunctivitis (Pinkeye)",
    "LSD": "Lumpy Skin Disease (Capripoxvirus)",
    "Normal": "Normal / Healthy (No Detected Pathology)"
}

DISEASE_EXPLANATION_MAP = {
    "FMD": "Foot-and-Mouth Disease is an acute, highly contagious viral disease causing vesicular lesions on the tongue, dental pad, gums, interdigital space, and teats, accompanied by excessive salivation, lameness, and sudden drop in milk yield.",
    "IBK": "Infectious Bovine Keratoconjunctivitis (Pinkeye) is a painful ocular bacterial infection caused primarily by Moraxella bovis, leading to blepharospasm, conjunctivitis, corneal opacity (cloudiness), ulceration, and potential blindness.",
    "LSD": "Lumpy Skin Disease is a capripoxviral infection characterized by firm, raised circumscribed skin nodules (2-5 cm), enlarged superficial lymph nodes, edema of limbs, ocular/nasal discharge, and marked production loss.",
    "Normal": "No clinical lesions, vesicles, corneal opacity, or cutaneous nodules detected. The anatomical presentation conforms to healthy bovine physiology."
}


class DiseaseInferenceService:
    """Service handling cattle disease vision diagnosis and vaccine guidance."""

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
        Diagnose cattle disease from image bytes and provide vaccine/timing/cost guidance.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ImageProcessingError(f"Failed to decode image: {str(e)}")

        model, classes = model_loader.load_disease_model()
        device = model_loader.device

        try:
            tensor = self.transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
            del tensor
            del logits

            max_idx = max(range(len(classes)), key=probabilities.__getitem__)
            predicted_class = classes[max_idx]
            confidence = float(probabilities[max_idx])
            prob_dict = {cls_name: float(prob) for cls_name, prob in zip(classes, probabilities)}

            is_disease = predicted_class != "Normal"
            full_name = DISEASE_NAME_MAP.get(predicted_class, predicted_class)
            explanation = DISEASE_EXPLANATION_MAP.get(predicted_class, "Clinical evaluation of bovine image.")

            # Retrieve vaccine, timing, and cost information from vaccination service
            vaccine_info = vaccination_service.get_vaccine_info_for_disease(predicted_class)

            return DiseasePredictionResponse(
                predicted_class=predicted_class,
                confidence=round(confidence, 4),
                confidence_percentage=round(confidence * 100.0, 2),
                is_disease_detected=is_disease,
                disease_name_full=full_name,
                explanation=explanation,
                recommended_vaccine=vaccine_info["recommended_vaccine"],
                vaccination_timing=vaccine_info["vaccination_timing"],
                estimated_cost=vaccine_info["estimated_cost"],
                brand_name=vaccine_info.get("brand_name"),
                manufacturer=vaccine_info.get("manufacturer"),
                price_type=vaccine_info.get("price_type"),
                farmer_cost_display=vaccine_info.get("farmer_cost_display"),
                calculated_per_dose_inr=vaccine_info.get("calculated_per_dose_inr"),
                procurement_cost_display=vaccine_info.get("procurement_cost_display"),
                retail_price_display=vaccine_info.get("retail_price_display"),
                source_name=vaccine_info.get("source_name"),
                source_url=vaccine_info.get("source_url"),
                source_date=vaccine_info.get("source_date"),
                is_stale=vaccine_info.get("is_stale", False),
                eligibility_notes=vaccine_info.get("eligibility_notes"),
                price_detail=vaccine_info.get("price_detail"),
                probabilities=prob_dict,
                model_version="efficientnet_b3",
                device_used=str(device),
                disclaimer="Screening tool for early detection. Consult a certified veterinarian for clinical treatment.",
                veterinary_disclaimer=vaccine_info["veterinary_disclaimer"]
            )

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ModelInferenceError("cattle_disease", str(e))


disease_service = DiseaseInferenceService()
