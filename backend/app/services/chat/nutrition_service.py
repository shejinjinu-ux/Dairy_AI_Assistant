"""
Field Nutrition & Least-Cost Ration Recommendation Service
Integrated with the Deterministic ICAR-NIANP Optimization Engine
"""

import abc
import re
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.nutrition import (
    NutritionRecommendationRequest,
    NutritionRecommendationResponse
)
from backend.app.services.nutrition_engine import nutrition_engine


class RationRequestModel(BaseModel):
    """Data model holding all farmer bovine nutrition parameters."""
    species: str = Field(default="Cattle", description="Bovine species ('Cattle' or 'Buffalo').")
    breed: Optional[str] = Field(default=None, description="Bovine breed (e.g., 'Gir', 'Sahiwal', 'Murrah', 'HF_Cross').")
    age_months: Optional[float] = Field(default=None, description="Age of animal in months.")
    body_weight_kg: Optional[float] = Field(default=None, description="Live body weight in kg.")
    lactation_stage: Optional[str] = Field(default=None, description="Lactation phase: 'Early', 'Mid', 'Late', 'Dry'.")
    days_in_milk: Optional[float] = Field(default=None, description="Days in current milk cycle.")
    daily_milk_yield_litres: Optional[float] = Field(default=None, description="Current daily milk yield in litres or kg.")
    milk_fat_percentage: Optional[float] = Field(default=None, description="Milk fat percentage (e.g. 4.0%). None if omitted.")
    pregnancy_status: Optional[bool] = Field(default=None, description="Whether the animal is pregnant. None if omitted.")
    pregnancy_month: Optional[int] = Field(default=None, description="Pregnancy month if pregnant.")
    available_feeds: List[str] = Field(default_factory=list, description="Available green, dry, and concentrate feeds on the farm.")
    feed_prices: Dict[str, float] = Field(default_factory=dict, description="Custom feed prices in INR/kg.")


class RationRecommendationResult(BaseModel):
    """Output contract for ration recommendation."""
    is_model_predicted: bool = Field(default=True, description="True when calculated by the ICAR-NIANP optimization engine or registered model.")
    status: str = Field(default="optimized", description="Status code ('optimized', 'missing_parameters', 'infeasible').")
    extracted_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters parsed from user message.")
    missing_critical_parameters: List[str] = Field(default_factory=list, description="Missing parameters needed for exact formulation.")
    general_guideline: str = Field(default="", description="Agronomic scientific feeding guideline.")
    recommendations: Optional[Dict[str, Any]] = Field(default=None, description="Structured output from ICAR optimizer or registered model.")
    formatted_summary: str = Field(default="", description="Human-readable formatted summary of optimal ration.")


class NutritionServiceInterface(abc.ABC):
    """Abstract Base Class / Service Interface for Field Nutrition."""

    @abc.abstractmethod
    def is_model_available(self) -> bool:
        """Returns True if an external ML model is registered."""
        pass

    @abc.abstractmethod
    def parse_nutrition_entities(self, text: str) -> RationRequestModel:
        """Extracts recognizable nutrition variables from natural text."""
        pass

    @abc.abstractmethod
    def generate_ration_advisory(
        self,
        text: str,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RationRecommendationResult:
        """Processes query and produces least-cost ration recommendation or parameter clarification."""
        pass

    @abc.abstractmethod
    def register_ml_model(self, model_callable: Callable[[RationRequestModel], Dict[str, Any]]) -> None:
        """Extensible hook for future external model registration."""
        pass


class DefaultNutritionService(NutritionServiceInterface):
    """
    Production-grade Nutrition Service implementation.
    Grounded in the deterministic ICAR-2013/2024 standards and ICAR-NIANP Feed Composition Database.
    Never invents missing critical animal values.
    """

    def __init__(self):
        self._ml_model_callable: Optional[Callable[[RationRequestModel], Dict[str, Any]]] = None

    def is_model_available(self) -> bool:
        return self._ml_model_callable is not None

    def register_ml_model(self, model_callable: Callable[[RationRequestModel], Dict[str, Any]]) -> None:
        """Plugs in an external model callable if registered."""
        self._ml_model_callable = model_callable

    def parse_nutrition_entities(self, text: str) -> RationRequestModel:
        """
        Extracts species, body weight, milk yield, fat %, and feed terms from farmer text across scripts.
        Strictly avoids inventing or assigning default values for unmentioned parameters.
        """
        params = RationRequestModel()
        clean = text.lower()

        # 1. Species detection (Buffalo vs Cattle)
        if any(w in clean for w in ["buffalo", "murrah", "எருமை", "भैंस", "గేదె", "ಎಮ್ಮೆ", "പോത്ത്", "মহিষ"]):
            params.species = "Buffalo"

        # 2. Milk yield in litres / kg
        milk_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:l|litre|liter|litres|லிட்டர்|லி|लीटर|ली|లీటర్ల|లీటర్|ಲೀಟರ್|ലിറ്റർ|লিটার|kg\s+milk)",
            clean
        )
        if milk_match:
            try:
                params.daily_milk_yield_litres = float(milk_match.group(1))
            except ValueError:
                pass

        # 3. Body weight in kg
        weight_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:kg|kilo|kilogram|கிலோ|किलो|కిలో|ಕಿಲೋ|കിലോ|কেজি)",
            clean
        )
        if weight_match:
            try:
                val = float(weight_match.group(1))
                if 120 <= val <= 1200:
                    params.body_weight_kg = val
            except ValueError:
                pass
        else:
            prefix_match = re.search(
                r"(?:weight|எடை|वजन|ತೂಕ|బరువు|ഭാരം|ওজন|weighs|is)\s*(?:is|:|=|ஆக)?\s*(\d+(?:\.\d+)?)",
                clean
            )
            if prefix_match:
                try:
                    val = float(prefix_match.group(1))
                    if 120 <= val <= 1200:
                        params.body_weight_kg = val
                except ValueError:
                    pass

        # 4. Milk fat percentage
        # A. Explicit fat keyword with number
        fat_kw_match = re.search(
            r"(?:fat|milk\s*fat|கொழுப்பு|फैट|ఫ్యాట్|వెన్న|కొవ్వు|ಕೊಬ್ಬು|കൊഴുപ്പ്|ফ্যাট|ਚਰਬੀ|स्निग्धांश|ফাট)\s*(?:is|:|=|ஆக|का|की)?\s*(\d+(?:\.\d+)?)\s*(?:%|pct|percent)?",
            clean
        )
        if fat_kw_match:
            try:
                f_val = float(fat_kw_match.group(1))
                if 2.0 <= f_val <= 15.0:
                    params.milk_fat_percentage = f_val
            except ValueError:
                pass

        # B. Percentage notation (e.g., "4%", "4.5 %", "4 percent")
        if params.milk_fat_percentage is None:
            pct_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|pct|percent)", clean)
            for p_str in pct_matches:
                try:
                    p_val = float(p_str)
                    if 2.5 <= p_val <= 14.0:
                        params.milk_fat_percentage = p_val
                        break
                except ValueError:
                    pass

        # C. Bare number if user just replied with number (e.g. "4" or "4.5" in a multi-turn clarification)
        if params.milk_fat_percentage is None and re.fullmatch(r"^\s*(\d+(?:\.\d+)?)\s*$", clean):
            try:
                val = float(clean.strip())
                if 2.5 <= val <= 14.0:
                    params.milk_fat_percentage = val
            except ValueError:
                pass

        # 5. Lactation stage keywords
        if any(w in clean for w in ["early", "1st month", "2nd month", "ஆரம்ப", "शुरुआती", "మొదటి", "ಪ್ರಾರಂಭ"]):
            params.lactation_stage = "Early"
        elif any(w in clean for w in ["mid", "மத்திம", "मध्य"]):
            params.lactation_stage = "Mid"
        elif any(w in clean for w in ["late", "dry", "கடைசி", "வற்றிய", "अंतिम", "చివరి"]):
            params.lactation_stage = "Late"

        # 6. Pregnancy status
        if any(w in clean for w in ["pregnant", "சினை", "गाभिन", "గర్భం", "ಗರ್ಭಿಣಿ", "ഗർഭിണി"]):
            params.pregnancy_status = True
            p_month = re.search(r"(\d+)\s*(?:month|மாத|महीने|నెల|తిಂಗಳ)", clean)
            if p_month:
                try:
                    params.pregnancy_month = int(p_month.group(1))
                except ValueError:
                    params.pregnancy_month = 7

        # 7. Available feed mentions
        feeds = []
        feed_map = {
            "napier": "Hybrid Napier (CO-3 / CO-4 / CO-5)",
            "நேப்பியர்": "Hybrid Napier (CO-3 / CO-4 / CO-5)",
            "नेपियर": "Hybrid Napier (CO-3 / CO-4 / CO-5)",
            "maize": "Maize Fodder (Green)",
            "மக்காச்சோளம்": "Maize Fodder (Green)",
            "मक्का": "Maize Fodder (Green)",
            "sorghum": "Sorghum / Jowar Fodder (Green)",
            "ஜோவர்": "Sorghum / Jowar Fodder (Green)",
            "ज्वार": "Sorghum / Jowar Fodder (Green)",
            "berseem": "Berseem / Egyptian Clover (Green)",
            "बरसीम": "Berseem / Egyptian Clover (Green)",
            "lucerne": "Lucerne / Alfalfa (Medicago sativa)",
            "குதிரை மசால்": "Lucerne / Alfalfa (Medicago sativa)",
            "paddy straw": "Paddy Straw (Oryza sativa)",
            "straw": "Paddy Straw (Oryza sativa)",
            "வைக்கோல்": "Paddy Straw (Oryza sativa)",
            "wheat straw": "Wheat Straw (Triticum aestivum)",
            "भूसा": "Wheat Straw (Triticum aestivum)",
            "silage": "Maize Silage (Fermented)",
            "சைலேஜ்": "Maize Silage (Fermented)",
            "साइलेज": "Maize Silage (Fermented)",
            "cottonseed": "Cottonseed Cake (Decorticated)",
            "பருத்தி கொட்டை": "Cottonseed Cake (Decorticated)",
            "बिनौला": "Cottonseed Cake (Decorticated)",
            "mustard cake": "Mustard / Rapeseed Cake (Expeller)",
            "सरसों खल": "Mustard / Rapeseed Cake (Expeller)",
            "groundnut cake": "Groundnut / Peanut Cake (Expeller)",
            "கடலை பிண்ணாக்கு": "Groundnut / Peanut Cake (Expeller)",
            "wheat bran": "Wheat Bran (Chokar)",
            "தவிடு": "Wheat Bran (Chokar)",
            "चोकर": "Wheat Bran (Chokar)",
            "dorb": "De-Oiled Rice Bran (DORB)",
            "compound feed": "Compound Cattle Feed (BIS Type-II Standard)",
            "அடர்தீவனம்": "Compound Cattle Feed (BIS Type-II Standard)",
            "दाना": "Compound Cattle Feed (BIS Type-II Standard)",
            "mineral": "Area Specific Mineral Mixture (ASMM)",
            "தாது உப்பு": "Area Specific Mineral Mixture (ASMM)",
            "खनिज मिश्रण": "Area Specific Mineral Mixture (ASMM)"
        }
        for kw, feed_name in feed_map.items():
            if kw in clean and feed_name not in feeds:
                feeds.append(feed_name)
        params.available_feeds = feeds

        return params

    def generate_ration_advisory(
        self,
        text: str,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        selected_cattle: Optional[Any] = None,
        analysis_records: Optional[List[Any]] = None
    ) -> RationRecommendationResult:
        """
        Processes natural text input, extracts animal parameters across conversation history,
        and binds authorized selected_cattle metrics if provided.
        Strictly requires body weight, milk yield, and milk fat % for lactating cows.
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

        extracted = self.parse_nutrition_entities(combined_text)

        # Bind metrics from authorized selected_cattle record if not explicitly specified in query
        if selected_cattle is not None:
            if extracted.body_weight_kg is None and hasattr(selected_cattle, "body_weight_kg"):
                extracted.body_weight_kg = selected_cattle.body_weight_kg
            if extracted.daily_milk_yield_litres is None and hasattr(selected_cattle, "daily_milk_yield_litres"):
                extracted.daily_milk_yield_litres = selected_cattle.daily_milk_yield_litres
            if extracted.milk_fat_percentage is None and hasattr(selected_cattle, "milk_fat_percentage"):
                extracted.milk_fat_percentage = selected_cattle.milk_fat_percentage
            if hasattr(selected_cattle, "species") and selected_cattle.species:
                extracted.species = selected_cattle.species
            if hasattr(selected_cattle, "breed") and selected_cattle.breed:
                extracted.breed = selected_cattle.breed
            if extracted.lactation_stage is None and hasattr(selected_cattle, "lactation_stage"):
                extracted.lactation_stage = selected_cattle.lactation_stage

        extracted_dict = extracted.model_dump(exclude_none=True)

        # Check if external ML model has been plugged in
        if self.is_model_available() and self._ml_model_callable:
            try:
                ml_res = self._ml_model_callable(extracted)
                return RationRecommendationResult(
                    is_model_predicted=True,
                    status="optimized",
                    extracted_parameters=extracted_dict,
                    missing_critical_parameters=[],
                    general_guideline="Generated via Registered Field Nutrition Model.",
                    recommendations=ml_res,
                    formatted_summary=str(ml_res)
                )
            except Exception:
                pass

        # Identify missing critical inputs for deterministic ICAR engine
        missing = []
        if extracted.body_weight_kg is None:
            missing.append("body_weight_kg")
        if extracted.daily_milk_yield_litres is None:
            missing.append("daily_milk_yield_litres")
        elif extracted.daily_milk_yield_litres > 0 and extracted.milk_fat_percentage is None:
            missing.append("milk_fat_percentage")

        if missing:
            # Build specific guideline based on missing parameters
            if missing == ["milk_fat_percentage"]:
                guideline = "Please provide your cow's milk fat percentage (milk fat %) to accurately calculate the balanced ration."
            else:
                guideline = (
                    "To formulate an optimal least-cost balanced ration (ICAR Standards), "
                    "please provide your animal's live body weight (kg), daily milk yield (litres), and milk fat percentage."
                )
            return RationRecommendationResult(
                is_model_predicted=False,
                status="missing_parameters",
                extracted_parameters=extracted_dict,
                missing_critical_parameters=missing,
                general_guideline=guideline,
                recommendations=None
            )

        # Build optimization request
        req = NutritionRecommendationRequest(
            species=extracted.species,
            breed=extracted.breed,
            body_weight_kg=extracted.body_weight_kg,
            daily_milk_yield_kg=extracted.daily_milk_yield_litres,
            milk_fat_percent=extracted.milk_fat_percentage,
            pregnancy_status=extracted.pregnancy_status,
            pregnancy_month=extracted.pregnancy_month,
            available_feeds=extracted.available_feeds if extracted.available_feeds else None,
            feed_prices=extracted.feed_prices if extracted.feed_prices else None
        )

        opt_response: NutritionRecommendationResponse = nutrition_engine.optimize_ration(req)

        if not opt_response.success:
            return RationRecommendationResult(
                is_model_predicted=False,
                status=opt_response.status,
                extracted_parameters=extracted_dict,
                missing_critical_parameters=opt_response.missing_critical_parameters,
                general_guideline=opt_response.message,
                recommendations=opt_response.model_dump()
            )

        # Build formatted summary string
        summary_lines = [
            f"Optimal Balanced Ration (ICAR-2013/2024 Standards | Total Cost: Rs.{opt_response.total_daily_cost_inr:.2f}/day):"
        ]
        for item in opt_response.recommended_ration:
            summary_lines.append(
                f"- {item.feed_name}: {item.quantity_kg_per_day:.2f} kg/day (Rs.{item.daily_cost_inr:.2f})"
            )

        nb = opt_response.nutrient_balance
        if nb:
            dm_stat = nb.get("dry_matter")
            tdn_stat = nb.get("tdn")
            cp_stat = nb.get("crude_protein")
            ca_stat = nb.get("calcium")
            p_stat = nb.get("phosphorus")
            summary_lines.append(
                f"Nutrient Fulfillment: DM {opt_response.nutrient_supply.get('dry_matter_kg', 0):.1f}kg ({dm_stat.percentage_fulfilled if dm_stat else 100}%), "
                f"TDN {opt_response.nutrient_supply.get('tdn_kg', 0):.2f}kg ({tdn_stat.percentage_fulfilled if tdn_stat else 100}%), "
                f"Protein {opt_response.nutrient_supply.get('crude_protein_g', 0):.0f}g ({cp_stat.percentage_fulfilled if cp_stat else 100}%), "
                f"Ca {opt_response.nutrient_supply.get('calcium_g', 0):.0f}g ({ca_stat.percentage_fulfilled if ca_stat else 100}%), "
                f"P {opt_response.nutrient_supply.get('phosphorus_g', 0):.0f}g ({p_stat.percentage_fulfilled if p_stat else 100}%)."
            )

        formatted_summary = "\n".join(summary_lines)

        return RationRecommendationResult(
            is_model_predicted=True,
            status="optimized",
            extracted_parameters=extracted_dict,
            missing_critical_parameters=[],
            general_guideline=opt_response.message,
            recommendations=opt_response.model_dump(),
            formatted_summary=formatted_summary
        )


# Global singleton nutrition service instance
nutrition_service = DefaultNutritionService()
