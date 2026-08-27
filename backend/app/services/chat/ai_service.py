"""
AI Response Generation & LLM Provider Abstraction Service
Combines high-precision built-in domain intelligence with optional external LLM providers
"""

import logging
from typing import Any, Dict, List, Optional
import httpx

from backend.config import settings
from backend.app.services.chat.knowledge_base import get_localized_response
from backend.app.services.chat.nutrition_service import RationRecommendationResult

logger = logging.getLogger("dairy_ai.chat.ai_service")


class AIService:
    """Production AI Response Generation Service."""

    def __init__(self):
        self.provider = settings.AI_PROVIDER.lower() if settings.AI_PROVIDER else "local"
        self.api_key = settings.AI_API_KEY
        self.model_name = settings.AI_MODEL

    def generate_response(
        self,
        user_message: str,
        target_language: str,
        intent: str,
        module: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        silage_data: Optional[Dict[str, Any]] = None,
        nutrition_data: Optional[RationRecommendationResult] = None
    ) -> str:
        """
        Generates farmer-friendly response in the target language.
        Prioritizes structured module insights (Silage / Nutrition) and local domain rules,
        with optional external LLM enhancement if configured.
        """
        # 1. Silage Module Specific Response Formatting
        if intent == "silage_quality" or module == "silage":
            if silage_data:
                # Silage model produced results
                fqi_val = silage_data.get("fermentation_quality_index", {}).get("predicted_fqi", 0.0)
                quality_cls = silage_data.get("quality_classification", {}).get("predicted_class", "ea")
                
                if fqi_val >= 75 or quality_cls == "ea":
                    base_reply = get_localized_response("silage_quality", target_language, "silage_good")
                    return f"{base_reply} (FQI Score: {fqi_val}/100)"
                else:
                    base_reply = get_localized_response("silage_quality", target_language, "silage_poor")
                    return f"{base_reply} (FQI Score: {fqi_val}/100)"
            else:
                # Missing test parameters -> Ask farmer for pH / moisture
                return get_localized_response("silage_quality", target_language, "silage_missing")

        # 2. Nutrition Module Specific Response Formatting
        if intent == "nutrition" or module == "nutrition":
            if nutrition_data and nutrition_data.status == "optimized" and nutrition_data.recommendations:
                recs = nutrition_data.recommendations
                total_cost = recs.get("total_daily_cost_inr", 0.0)
                items = recs.get("recommended_ration", [])
                
                # Multilingual Headers
                headers = {
                    "en": f"Optimal Balanced Ration (ICAR-2013/2024 Standards | Total Cost: Rs.{total_cost:.2f}/day):",
                    "ta": f"சீரான தீவன பரிந்துரை (ICAR தரநிலை | தினசரி செலவு: ரூ.{total_cost:.2f}/நாள்):",
                    "hi": f"संतुलित आहार सिफारिश (ICAR मानक | दैनिक लागत: रु.{total_cost:.2f}/दिन):",
                    "te": f"సమతుల్య ఆహార ప్రణాళిక (ICAR ప్రమాణాలు | రోజువారీ ఖర్చు: రూ.{total_cost:.2f}/రోజు):",
                    "kn": f"ಸಮತೋಲಿತ ಆಹಾರ ಶಿಫಾರಸು (ICAR ಮಾನದಂಡಗಳು | ದಿನದ ವೆಚ್ಚ: ರೂ.{total_cost:.2f}/ದಿನ):",
                    "ml": f"സമീകൃത തീറ്റക്രമം (ICAR നിലവാരം | പ്രതിദിന ചെലവ്: രൂ.{total_cost:.2f}/ദിവസം):",
                    "bn": f"সুষম খাদ্য তালিকা (ICAR মানক | দৈনিক খরচ: টাকা.{total_cost:.2f}/দিন):",
                    "mr": f"संतुलित आहार शिफारस (ICAR मानके | दैनिक खर्च: रु.{total_cost:.2f}/दिवस):",
                    "gu": f"સંતુલિત આહાર ભલામણ (ICAR ધોરણો | દૈનિક ખર્ચ: રૂ.{total_cost:.2f}/દિવસ):",
                    "pa": f"ਸੰਤੁਲਿਤ ਖੁਰਾਕ ਦੀ ਸਿਫਾਰਸ਼ (ICAR ਮਿਆਰ | ਰੋਜ਼ਾਨਾ ਖਰਚ: ਰੁ.{total_cost:.2f}/ਦਿਨ):",
                    "or": f"ସନ୍ତୁଳିତ ଆହାର ସୁପାରିଶ (ICAR ମାନକ | ଦୈନିକ ଖର୍ଚ୍ଚ: ଟଙ୍କା.{total_cost:.2f}/ଦିନ):",
                    "as": f"সুষম আহাৰৰ পৰামৰ্শ (ICAR মানদণ্ড | দৈনন্দিন খৰচ: টকা.{total_cost:.2f}/দিন):",
                    "ur": f"متوازن خوراک کی سفارش (ICAR معیارات | یومیہ لاگت: روپے {total_cost:.2f}/دن):"
                }
                header_text = headers.get(target_language, f"Optimal Balanced Ration (ICAR Standards | Total Cost: Rs.{total_cost:.2f}/day):")
                
                lines = [header_text]
                for itm in items:
                    fname = itm.get("feed_name", "")
                    qty = itm.get("quantity_kg_per_day", 0.0)
                    icost = itm.get("daily_cost_inr", 0.0)
                    lines.append(f"• {fname}: {qty:.2f} kg/day (Rs.{icost:.2f})")

                sup = recs.get("nutrient_supply", {})
                dm_val = sup.get("dry_matter_kg", 0.0)
                cp_val = sup.get("crude_protein_g", 0.0)
                tdn_val = sup.get("tdn_kg", 0.0)
                lines.append(f"Total Nutrient Supply: Dry Matter {dm_val:.1f} kg, Energy (TDN) {tdn_val:.2f} kg, Crude Protein {cp_val:.0f} g.")
                return "\n".join(lines)
            else:
                # Missing parameters -> Check specifically which parameters are missing
                missing_list = nutrition_data.missing_critical_parameters if nutrition_data else []
                if missing_list == ["milk_fat_percentage"] or missing_list == ["milk_fat_percent"]:
                    fat_prompts = {
                        "en": "Please provide your cow's milk fat percentage (milk fat %) to accurately calculate the balanced ration.",
                        "ta": "உங்கள் மாட்டின் பால் கொழுப்பு சதவீதம் (milk fat %) எவ்வளவு என்று கூறவும்.",
                        "hi": "कृपया अपनी गाय के दूध में फैट प्रतिशत (milk fat %) बताएं ताकि सटीक संतुलित आहार की गणना की जा सके।",
                        "te": "దయచేసి మీ పశువు పాలల్లో వెన్న/కొవ్వు శాతం (milk fat %) తెలపండి.",
                        "kn": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹಸುವಿನ ಹಾಲಿನ ಕೊಬ್ಬಿನ ಪ್ರಮಾಣವನ್ನು (milk fat %) ತಿಳಿಸಿ.",
                        "ml": "പാലിന്റെ കൊഴുപ്പ് ശതമാനം (milk fat %) എത്രയാണെന്ന് വ്യക്തമാക്കുക.",
                        "bn": "সঠিক খাদ্য তালিকা নির্ধারণ করতে দুধের ফ্যাট শতাংশ (milk fat %) উল্লেখ করুন।",
                        "mr": "कृपया दुधातील फॅटचे प्रमाण (milk fat %) सांगा.",
                        "gu": "કૃપા કરીને દૂધમાં ફેટની ટકાવારી (milk fat %) જણાવો.",
                        "pa": "ਕਿਰਪਾ ਕਰਕੇ ਦੁੱਧ ਵਿੱਚ ਫੈਟ ਦੀ ਪ੍ਰਤੀਸ਼ਤਤਾ (milk fat %) ਦੱਸੋ।",
                        "or": "ଦୟାକରି କ୍ଷୀରର ଫ୍ୟାଟ୍ ପ୍ରତିଶତ (milk fat %) ଜଣାନ୍ତୁ।",
                        "as": "অনুগ্ৰহ কৰি গাখীৰত চৰ্বিৰ শতাংশ (milk fat %) উল্লেখ কৰক।",
                        "ur": "براہ کرم متوازن راشن کے حساب کے لیے دودھ میں چکنائی کا تناسب (milk fat %) بتائیں۔"
                    }
                    fat_msg = fat_prompts.get(target_language, "Please provide your cow's milk fat percentage (milk fat %) to accurately calculate the balanced ration.")
                    return fat_msg
                else:
                    missing_msg = get_localized_response("nutrition", target_language, "nutrition_missing")
                    general_rule = get_localized_response("nutrition", target_language, "nutrition_general")
                    return f"{missing_msg}\n\n{general_rule}"


        # 3. Dedicated Intent Responses (Greeting, Health, Feed, Milk Production, General Dairy)
        if intent in ["greeting", "feed", "cattle_health_general", "milk_production", "animal_profile", "general_dairy", "unknown"]:
            return get_localized_response(intent, target_language)

        # 4. Optional External LLM Integration (Gemini / OpenAI) if key configured
        if self.api_key and self.provider != "local":
            try:
                llm_reply = self._call_external_llm(user_message, target_language, conversation_history)
                if llm_reply:
                    return llm_reply
            except Exception as e:
                logger.warning(f"External LLM call failed: {e}. Falling back to local engine.")

        # Default fallback
        return get_localized_response("unknown", target_language)

    def _call_external_llm(
        self,
        message: str,
        language: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """Calls external Gemini / OpenAI API if configured."""
        if not self.api_key:
            return None

        # Gemini REST API implementation
        if "gemini" in self.provider or "google" in self.provider:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name or 'gemini-1.5-flash'}:generateContent?key={self.api_key}"
            prompt = (
                f"You are a helpful Dairy AI Assistant for farmers. Reply concisely and warmly in language '{language}'. "
                f"Preserve all quantities and units. Avoid medical diagnosis; urge vet consultation for sick cattle.\n"
                f"Farmer message: {message}"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 300}
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        return candidates[0]["content"]["parts"][0]["text"].strip()
        return None


ai_service = AIService()
