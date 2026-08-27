"""
Silage Chat Integration Service
Integrates with the existing verified Silage inference engine and provides farmer-friendly silage advice
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from backend.app.schemas.silage import SilageInput

from backend.app.services.silage_service import silage_service


class SilageChatService:
    """Handles chat queries regarding silage quality and fermentation."""

    def extract_silage_parameters(self, text: str) -> Dict[str, float]:
        """Extracts any explicitly provided proximal/fermentation numerical values."""
        params: Dict[str, float] = {}
        clean = text.lower()

        # Extract pH (e.g., "pH 3.8", "ph is 4.2", "ph=3.9")
        ph_match = re.search(r"\bph(?:\s+is|\s*[:=])?\s*(\d+(?:\.\d+)?)\b", clean)
        if ph_match:
            try:
                params["pH"] = float(ph_match.group(1))
            except ValueError:
                pass

        # Extract Moisture / Dry matter
        dm_match = re.search(r"(?:dry\s*matter|dm|உலர்\s*பொருள்|शुष्क\s*पदार्थ)(?:\s+is|\s*[:=])?\s*(\d+(?:\.\d+)?)\s*%?", clean)
        if dm_match:
            try:
                params["dm.s"] = float(dm_match.group(1))
            except ValueError:
                pass

        moisture_match = re.search(r"(?:moisture|ஈரப்பதம்|नमी)(?:\s+is|\s*[:=])?\s*(\d+(?:\.\d+)?)\s*%?", clean)
        if moisture_match:
            try:
                moisture_val = float(moisture_match.group(1))
                # DM = 100 - Moisture
                if "dm.s" not in params:
                    params["dm.s"] = round(100.0 - moisture_val, 1)
            except ValueError:
                pass

        # Extract crude protein (CP)
        cp_match = re.search(r"(?:crude\s*protein|cp|புரதம்|प्रोटीन)(?:\s+is|\s*[:=])?\s*(\d+(?:\.\d+)?)\s*%?", clean)
        if cp_match:
            try:
                params["cp.s"] = float(cp_match.group(1))
            except ValueError:
                pass

        return params

    def evaluate_silage_query(self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, float]]:
        """
        Determines whether sufficient parameters exist to run the production XGBoost model,
        or if missing critical parameters should be requested from the farmer.
        Supports multi-turn parameter aggregation across conversation history.
        """
        combined_text = text
        if conversation_history:
            past_texts = [
                m.get("message") or m.get("content", "")
                for m in conversation_history
                if m.get("role") == "user"
            ]
            if past_texts:
                combined_text = " ".join(past_texts) + " " + text

        extracted = self.extract_silage_parameters(combined_text)



        # Check if enough parameters to construct full SilageInput
        # If user supplied full/key fields like pH and DM
        if "pH" in extracted and "dm.s" in extracted:
            try:
                # Build SilageInput using standard agronomic baseline enriched with extracted parameters
                input_payload = SilageInput(
                    pH=extracted["pH"],
                    dm_s=extracted["dm.s"],
                    cp_s=extracted.get("cp.s", 14.0)
                )
                result = silage_service.predict_comprehensive(input_payload)
                return True, result.model_dump(), extracted
            except Exception:
                pass

        return False, None, extracted


silage_chat_service = SilageChatService()
