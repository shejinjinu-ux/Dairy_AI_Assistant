"""
AI Response Generation & LLM Provider Abstraction Service
Combines high-precision built-in domain intelligence with dynamic Google Gemini API integration
"""

import re
import logging
from typing import Any, Dict, List, Optional
import httpx

from backend.config import settings
from backend.app.services.chat.knowledge_base import get_localized_response
from backend.app.services.chat.nutrition_service import RationRecommendationResult
from backend.app.schemas.feed_reference import FeedReferenceRequest
from backend.app.services.feed_reference_service import feed_reference_service

logger = logging.getLogger("dairy_ai.chat.ai_service")

# Human-readable language map for Gemini system instructions
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "ta": "Tamil (தமிழ்)",
    "hi": "Hindi (हिन्दी)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "bn": "Bengali (বাংলা)",
    "mr": "Marathi (मराठी)",
    "gu": "Gujarati (ગુજરાતી)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "or": "Odia (ଓଡ଼ିଆ)",
    "as": "Assamese (অসমীয়া)",
    "ur": "Urdu (اردو)",
    "sa": "Sanskrit (संस्कृतम्)",
    "ne": "Nepali (नेपाली)",
    "kok": "Konkani (कोंकणी)",
    "mai": "Maithili (मैथिली)",
    "mni": "Manipuri (ꯃꯤꯇꯩꯂꯣꯟ)",
    "ks": "Kashmiri (کٲشُر)",
    "sd": "Sindhi (سنڌي)",
    "doi": "Dogri (डोगरी)",
    "sat": "Santali (ᱥᱟᱱᱛᱟᱲᱤ)",
    "brx": "Bodo (बर')",
}


class AIService:
    """Production AI Response Generation Service supporting Google Gemini and Local Engines."""

    def __init__(self, http_client: Optional[httpx.Client] = None):
        self._http_client = http_client
        self.last_provider: str = "local"
        self.last_model_used: Optional[str] = None
        self.last_error: Optional[str] = None

    @property
    def provider(self) -> str:
        return (settings.AI_PROVIDER or "local").lower().strip()

    @property
    def api_key(self) -> Optional[str]:
        return settings.effective_ai_api_key

    @property
    def model_name(self) -> str:
        return settings.effective_ai_model

    @property
    def is_gemini_active(self) -> bool:
        return bool(settings.is_gemini_configured)

    def generate_response(
        self,
        user_message: str,
        target_language: str,
        intent: str,
        module: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        silage_data: Optional[Dict[str, Any]] = None,
        nutrition_data: Optional[RationRecommendationResult] = None,
        selected_cattle: Optional[Any] = None,
        analysis_records: Optional[List[Any]] = None
    ) -> str:
        """
        Generates farmer-friendly response in the target language.
        Prioritizes authorized selected cattle context and persistent analysis records.
        When Gemini is configured (AI_PROVIDER='gemini' and API key present), queries are dynamically
        answered by Gemini with multi-turn conversation memory and veterinary guidance.
        When local provider is active, routes through localized rule-based domain intelligence.
        """
        self.last_error = None

        # 1. Specialized Calculation Results (When actual numeric model evaluations were produced)
        # 1a. Silage Model Calculation Output
        if (intent == "silage_quality" or module == "silage") and silage_data:
            self.last_provider = "domain_silage"
            self.last_model_used = None
            fqi_val = silage_data.get("fermentation_quality_index", {}).get("predicted_fqi", 0.0)
            quality_cls = silage_data.get("quality_classification", {}).get("predicted_class", "ea")
            
            if fqi_val >= 75 or quality_cls == "ea":
                base_reply = get_localized_response("silage_quality", target_language, "silage_good")
                return f"{base_reply} (FQI Score: {fqi_val}/100)"
            else:
                base_reply = get_localized_response("silage_quality", target_language, "silage_poor")
                return f"{base_reply} (FQI Score: {fqi_val}/100)"

        # 1b. Nutrition Model Calculation Output
        if (intent == "nutrition" or module == "nutrition") and nutrition_data and nutrition_data.status == "optimized" and nutrition_data.recommendations:
            self.last_provider = "domain_nutrition"
            self.last_model_used = None
            return self._format_nutrition_table(nutrition_data, target_language)

        # 1c. Specific Feed Ingredient / Reference Nutrition Match
        if (intent == "feed" or "feed" in module):
            feed_ref_reply = self._check_feed_reference_nutrition(user_message, target_language)
            if feed_ref_reply:
                self.last_provider = "domain_feed_reference"
                self.last_model_used = None
                return feed_ref_reply

        # 2. Dynamic Gemini LLM Integration
        if self.is_gemini_active:
            try:
                gemini_reply = self._call_gemini(
                    message=user_message,
                    language=target_language,
                    history=conversation_history,
                    intent=intent,
                    module=module
                )
                if gemini_reply and len(gemini_reply.strip()) > 0:
                    self.last_provider = "gemini"
                    return gemini_reply.strip()
            except Exception as e:
                self.last_error = f"{type(e).__name__}"
                logger.warning(f"Gemini API invocation failed: {type(e).__name__}. Falling back to local engine.")

            self.last_provider = "local_fallback"
        else:
            self.last_provider = "local"
            self.last_model_used = None

        # 3. Local Built-in Domain Engine (Fallback or when AI_PROVIDER='local')
        return self._generate_local_response(
            user_message=user_message,
            target_language=target_language,
            intent=intent,
            module=module,
            nutrition_data=nutrition_data
        )

    def _call_gemini(
        self,
        message: str,
        language: str,
        history: Optional[List[Dict[str, Any]]] = None,
        intent: Optional[str] = None,
        module: Optional[str] = None
    ) -> Optional[str]:
        """
        Calls Google Gemini REST API with structured conversation history,
        multilingual prompt, context-specific guidelines, and model candidate fallbacks.
        """
        api_key = self.api_key
        if not api_key:
            return None

        lang_name = LANGUAGE_NAMES.get(language, language)

        # Context-aware system prompt
        system_instruction = (
            "You are Dairy Nova AI, an intelligent, helpful, and polite AI assistant for dairy farming and general knowledge in India.\n"
            "Core Guidelines:\n"
            f"1. Respond concisely, warmly, and helpfully in {lang_name} (language code: '{language}').\n"
            "2. For general or non-dairy queries (e.g. computer science, networking, coding, general science, geography, history, everyday conversation), "
            "answer directly, accurately, and naturally without forcing dairy context.\n"
            "3. For dairy farming, cattle care, bovine breeds, feed nutrition, silage, and milk production, provide practical, "
            "scientifically sound advice aligned with Indian dairy management and ICAR benchmarks.\n"
            "4. For cattle disease or health issues, provide informational guidance while advising the farmer to consult a local veterinarian for sick animals.\n"
            "5. Preserve all numeric quantities, units (kg, litres, %, etc.), and clear concise bullet points where appropriate."
        )

        # Build multi-turn message contents
        contents: List[Dict[str, Any]] = []

        if history:
            # Include up to the last 6 messages to preserve conversational continuity
            for msg in history[-6:]:
                role = "user" if msg.get("role") == "user" else "model"
                text = msg.get("message", "")
                if text and text.strip():
                    contents.append({"role": role, "parts": [{"text": text.strip()}]})

        # Append current user turn
        if not contents or contents[-1].get("role") != "user" or contents[-1].get("parts", [{}])[0].get("text") != message.strip():
            contents.append({"role": "user", "parts": [{"text": message.strip()}]})

        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "maxOutputTokens": 1024
            }
        }

        headers = {"Content-Type": "application/json"}

        # Build ordered candidate models list
        candidate_models = [self.model_name]
        for fallback_model in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview"]:
            if fallback_model not in candidate_models:
                candidate_models.append(fallback_model)

        for model_to_try in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_try}:generateContent?key={api_key}"
            try:
                res = self._send_gemini_request(url, headers, payload, timeout=15.0)

                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            self.last_model_used = model_to_try
                            return parts[0]["text"].strip()
                    logger.warning(f"Gemini returned empty candidates for model '{model_to_try}': {data}")
                elif res.status_code == 404:
                    # Model not found or sunset; try next fallback model candidate
                    logger.info(f"Gemini model '{model_to_try}' returned 404 NOT_FOUND. Trying next candidate...")
                    continue
                else:
                    logger.warning(f"Gemini API returned status code {res.status_code} for model '{model_to_try}'.")
                    # If rate limited or service error, try another flash candidate
                    continue
            except httpx.TimeoutException:
                logger.warning(f"Gemini API request timed out after 15.0s for model '{model_to_try}'.")
                continue
            except httpx.RequestError as exc:
                logger.warning(f"Network error connecting to Gemini API with model '{model_to_try}': {type(exc).__name__}")
                continue
            except Exception as exc:
                logger.warning(f"Unexpected error calling Gemini API with model '{model_to_try}': {type(exc).__name__}")
                continue

        return None

    def _send_gemini_request(self, url: str, headers: dict, payload: dict, timeout: float = 15.0) -> httpx.Response:
        """Isolated HTTP request execution for Google Gemini REST endpoint."""
        if self._http_client:
            return self._http_client.post(url, headers=headers, json=payload, timeout=timeout)
        with httpx.Client(timeout=timeout) as client:
            return client.post(url, headers=headers, json=payload)

    def _format_nutrition_table(self, nutrition_data: RationRecommendationResult, target_language: str) -> str:
        """Formats optimized ICAR nutrition recommendation into multilingual response."""
        recs = nutrition_data.recommendations
        total_cost = recs.get("total_daily_cost_inr", 0.0)
        items = recs.get("recommended_ration", [])

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

    def _check_feed_reference_nutrition(self, user_message: str, target_language: str) -> Optional[str]:
        """Checks for specific feed ingredient nutrition reference queries."""
        clean_msg_lower = user_message.lower()
        if any(q in clean_msg_lower for q in ["protein", "nutrition", "energy", "value", "content", "கச்சா புரதம்", "ஊட்டச்சத்து", "प्रोटीन", "पोषण"]):
            words = user_message.split()
            for token_len in [3, 2, 1]:
                for i in range(len(words) - token_len + 1):
                    sub = " ".join(words[i:i+token_len])
                    matched = feed_reference_service.find_feed(sub)
                    if matched:
                        qty_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:kg|kilo|litres?|l)?', user_message, re.IGNORECASE)
                        qty = float(qty_match.group(1)) if qty_match and float(qty_match.group(1)) < 200 else 1.0
                        ref_res = feed_reference_service.calculate_nutrition(
                            FeedReferenceRequest(feed_name=matched["feed_name"], quantity_kg=qty)
                        )
                        fname = ref_res.matched_feed_name
                        dm = ref_res.total_for_quantity.dry_matter_g
                        cp = ref_res.total_for_quantity.crude_protein_g
                        me = ref_res.total_for_quantity.energy_mj
                        
                        if target_language == "ta":
                            return f"{fname} ({qty} கிலோ): உலர் பொருள் {dm} கி, கச்சா புரதம் {cp} கி, ஆற்றல் {me} MJ (ICAR-NIANP தரவு)."
                        elif target_language == "hi":
                            return f"{fname} ({qty} किलो): शुष्क पदार्थ {dm} ग्राम, कच्चा प्रोटीन {cp} ग्राम, ऊर्जा {me} MJ (ICAR-NIANP संदर्भ)."
                        else:
                            return f"{fname} ({qty} kg): Dry Matter {dm}g, Crude Protein {cp}g, Energy {me} MJ (ICAR-NIANP reference composition)."
        return None

    def _generate_local_response(
        self,
        user_message: str,
        target_language: str,
        intent: str,
        module: str,
        nutrition_data: Optional[RationRecommendationResult] = None
    ) -> str:
        """Local domain knowledge base fallback."""
        if intent == "silage_quality" or module == "silage":
            return get_localized_response("silage_quality", target_language, "silage_missing")

        if intent == "nutrition" or module == "nutrition":
            missing_list = nutrition_data.missing_critical_parameters if nutrition_data else []
            if missing_list in [["milk_fat_percentage"], ["milk_fat_percent"]]:
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
                return fat_prompts.get(target_language, "Please provide your cow's milk fat percentage (milk fat %) to accurately calculate the balanced ration.")
            else:
                missing_msg = get_localized_response("nutrition", target_language, "nutrition_missing")
                general_rule = get_localized_response("nutrition", target_language, "nutrition_general")
                return f"{missing_msg}\n\n{general_rule}"

        if intent in ["greeting", "feed", "cattle_health_general", "milk_production", "animal_profile", "general_dairy"]:
            return get_localized_response(intent, target_language)

        return get_localized_response("unknown", target_language)


ai_service = AIService()
