"""
Vaccination Recommendation & Schedule Service
Provides source-backed Indian veterinary pricing, government programmes (NADCP ₹0),
institutional procurement benchmarks, per-dose math, stale price detection, and authoritative citations.
"""

import logging
from datetime import datetime, timezone, timedelta, date
from typing import Dict, List, Optional, Any

from backend.app.schemas.user_farm_cattle import (
    Cattle,
    VaccinationRecord,
    VaccinationRecommendation,
    VaccinePriceDetail
)

logger = logging.getLogger("dairy_ai.services.vaccination")

RETAIL_UNAVAILABLE_MESSAGE = "Retail price unavailable — check local veterinary pharmacy / Animal Husbandry Department."
UNAVAILABLE_PRICE_MESSAGE = RETAIL_UNAVAILABLE_MESSAGE

# Authoritative Source-Backed Indian Veterinary Vaccine Schedule & Price Catalogue
STANDARD_VACCINATION_SCHEDULE: Dict[str, Dict[str, Any]] = {
    "FMD": {
        "disease_name": "Foot-and-Mouth Disease (FMD)",
        "recommended_vaccine": "Inactivated Trivalent FMD Vaccine (Type O, A, Asia-1)",
        "brand_name": "Raksha-Ovac / Raksha-Biovac",
        "manufacturer": "Indian Immunologicals Ltd (IIL) / Brilliant Bio Pharma",
        "pack_size_doses": 100,
        "total_pack_price_inr": 1800.0,
        "calculated_per_dose_inr": 18.0,
        "procurement_cost_inr": 18.0,
        "procurement_cost_display": "₹18.00 / dose (Government Procurement Price)",
        "retail_price_inr": None,
        "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
        "price_type": "GOVERNMENT_PROGRAMME_FREE",
        "farmer_cost_inr": 0.0,
        "farmer_cost_display": "₹0 (Government Programme / Farmer Cost)",
        "cost_per_dose_display": "₹0 (Government Programme / Farmer Cost) | Procurement: ₹18.00 / dose",
        "state_market": "All India (NADCP)",
        "source_name": "Department of Animal Husbandry & Dairying (DAHD), Ministry of Fisheries, Animal Husbandry & Dairying",
        "source_url": "https://dahd.nic.in/schemes/programmes/nadcp",
        "source_date": "2024-01-15",
        "recommended_timing": "Primary vaccination at 4 months of age; bi-annual booster every 6 months (pre-monsoon & post-monsoon).",
        "interval_days": 180,
        "notes": "Administer subcutaneously (2 ml). Avoid vaccinating animals during late pregnancy.",
        "eligibility_notes": "100% centrally sponsored under National Animal Disease Control Programme (NADCP). Available free of cost for all cattle and buffaloes nationwide.",
    },
    "BRUCELLOSIS": {
        "disease_name": "Brucellosis (Contagious Abortion)",
        "recommended_vaccine": "Brucella abortus S19 Live Freeze-Dried Vaccine",
        "brand_name": "Bruvax",
        "manufacturer": "Indian Immunologicals Ltd (IIL)",
        "pack_size_doses": 10,
        "total_pack_price_inr": 220.0,
        "calculated_per_dose_inr": 22.0,
        "procurement_cost_inr": 22.0,
        "procurement_cost_display": "₹22.00 / dose (Government Procurement Price)",
        "retail_price_inr": None,
        "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
        "price_type": "GOVERNMENT_PROGRAMME_FREE",
        "farmer_cost_inr": 0.0,
        "farmer_cost_display": "₹0 (Government Programme / Farmer Cost for Calves)",
        "cost_per_dose_display": "₹0 (Government Programme / Farmer Cost for Calves) | Procurement: ₹22.00 / dose",
        "state_market": "All India (NADCP)",
        "source_name": "Department of Animal Husbandry & Dairying (DAHD) / NADCP Brucellosis Operational Guidelines",
        "source_url": "https://dahd.nic.in/schemes/programmes/nadcp",
        "source_date": "2024-01-15",
        "recommended_timing": "Single dose for female calves aged 4 to 8 months only.",
        "interval_days": None,  # Lifetime single dose
        "notes": "Do NOT vaccinate pregnant cows or adult male bulls. Strict zoonotic precaution during handling.",
        "eligibility_notes": "100% centrally sponsored under NADCP. Free one-time vaccination for female calves aged 4–8 months only.",
    },
    "HS": {
        "disease_name": "Haemorrhagic Septicaemia (HS)",
        "recommended_vaccine": "Pasteurella multocida Alum Precipitated / Oil Adjuvant Vaccine",
        "brand_name": "Raksha-HS / Bio-HS",
        "manufacturer": "Indian Immunologicals Ltd (IIL) / Brilliant Bio Pharma",
        "pack_size_doses": 50,
        "total_pack_price_inr": 650.0,
        "calculated_per_dose_inr": 13.0,
        "procurement_cost_inr": 13.0,
        "procurement_cost_display": "₹13.00 / dose (Government Procurement Price)",
        "retail_price_inr": None,
        "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
        "price_type": "GOVERNMENT_PROCUREMENT",
        "farmer_cost_inr": 0.0,
        "farmer_cost_display": "₹0 during State AH Department Campaigns",
        "cost_per_dose_display": "₹13.00 / dose (Government Procurement Price)",
        "state_market": "State Animal Husbandry Depts / GeM",
        "source_name": "State Animal Husbandry Department Annual Rate Contract & GeM Procurement Bulletin",
        "source_url": "https://gem.gov.in",
        "source_date": "2024-02-10",
        "recommended_timing": "Annual vaccination prior to monsoon onset (May-June).",
        "interval_days": 365,
        "notes": "Critical in flood-prone and low-lying agrarian districts. Subcutaneous route (2 ml).",
        "eligibility_notes": "Supplied free by State Animal Husbandry Departments during pre-monsoon vaccination campaigns in endemic areas.",
    },
    "BQ": {
        "disease_name": "Black Quarter (BQ)",
        "recommended_vaccine": "Polyvalent Clostridium chauvoei Inactivated Vaccine",
        "brand_name": "Raksha-BQ",
        "manufacturer": "Indian Immunologicals Ltd (IIL) / State Biological Production Units",
        "pack_size_doses": 50,
        "total_pack_price_inr": 600.0,
        "calculated_per_dose_inr": 12.0,
        "procurement_cost_inr": 12.0,
        "procurement_cost_display": "₹12.00 / dose (Government Procurement Price)",
        "retail_price_inr": None,
        "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
        "price_type": "GOVERNMENT_PROCUREMENT",
        "farmer_cost_inr": 0.0,
        "farmer_cost_display": "₹0 during State AH Department Campaigns",
        "cost_per_dose_display": "₹12.00 / dose (Government Procurement Price)",
        "state_market": "State Animal Husbandry Depts / GeM",
        "source_name": "Government e-Marketplace (GeM) / State Veterinary Biologicals Procurement",
        "source_url": "https://gem.gov.in",
        "source_date": "2023-11-20",
        "recommended_timing": "Annual vaccination in cattle aged 6 months to 3 years before monsoon (May-June).",
        "interval_days": 365,
        "notes": "Young stock (6-24 months) are most susceptible. Subcutaneous route (2 ml).",
        "eligibility_notes": "Supplied free through State Department of Animal Husbandry veterinary dispensaries in vulnerable blocks.",
    },
    "LSD": {
        "disease_name": "Lumpy Skin Disease (Capripoxvirus)",
        "recommended_vaccine": "Heterologous Live Goat Pox Vaccine (Uttarkashi Strain) / Lumpi-ProVacInd",
        "brand_name": "GPOX-MV / Raksha-LSD",
        "manufacturer": "Hester Biosciences / Indian Immunologicals Ltd (IIL) / ICAR-NRCE",
        "pack_size_doses": 100,
        "total_pack_price_inr": 1200.0,
        "calculated_per_dose_inr": 12.0,
        "procurement_cost_inr": 12.0,
        "procurement_cost_display": "₹12.00 / dose (Government Procurement Price)",
        "retail_price_inr": None,
        "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
        "price_type": "GOVERNMENT_PROCUREMENT",
        "farmer_cost_inr": 0.0,
        "farmer_cost_display": "₹0 during State Emergency Vaccination Drives",
        "cost_per_dose_display": "₹12.00 / dose (Government Procurement Price)",
        "state_market": "All India / State Animal Husbandry Emergency Drives",
        "source_name": "DAHD LSD Advisory & ICAR-NRCE Commercialization Guidelines",
        "source_url": "https://dahd.nic.in",
        "source_date": "2024-03-01",
        "recommended_timing": "Annual booster vaccination in healthy cattle & calves above 3 months.",
        "interval_days": 365,
        "notes": "Ensures herd immunity against vector-borne capripox transmission.",
        "eligibility_notes": "Provided free by state veterinary teams during outbreak containment drives.",
    },
    "ANTHRAX": {
        "disease_name": "Anthrax (Bacillus anthracis)",
        "recommended_vaccine": "Anthrax Spore Vaccine (Sterne Strain 34F2 Live)",
        "brand_name": "Anthrax Spore Vaccine (IVPM / VBRI)",
        "manufacturer": "State Veterinary Biological Research Institutes (IVPM Ranipet / VBRI)",
        "pack_size_doses": 50,
        "total_pack_price_inr": 400.0,
        "calculated_per_dose_inr": 8.0,
        "procurement_cost_inr": 8.0,
        "procurement_cost_display": "₹8.00 / dose (Government Procurement Price)",
        "retail_price_inr": None,
        "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
        "price_type": "GOVERNMENT_PROCUREMENT",
        "farmer_cost_inr": 0.0,
        "farmer_cost_display": "₹0 in Endemic Belts via State AH Dept",
        "cost_per_dose_display": "₹8.00 / dose (Government Procurement Price)",
        "state_market": "State Endemic Districts",
        "source_name": "State Institute of Veterinary Preventive Medicine Rate Schedule & DAHD",
        "source_url": "https://dahd.nic.in",
        "source_date": "2023-10-15",
        "recommended_timing": "Annual vaccination in known endemic districts only (pre-monsoon).",
        "interval_days": 365,
        "notes": "Administer strictly in designated endemic belts under direct veterinary supervision.",
        "eligibility_notes": "100% state-supplied free vaccination in declared anthrax-endemic revenue villages.",
    },
    "THEILERIOSIS": {
        "disease_name": "Bovine Theileriosis (Theileria annulata)",
        "recommended_vaccine": "Rakshavac-T / Cell Culture Schizont Live Vaccine",
        "brand_name": "Rakshavac-T",
        "manufacturer": "Indian Immunologicals Ltd (IIL)",
        "pack_size_doses": None,
        "total_pack_price_inr": None,
        "calculated_per_dose_inr": None,
        "procurement_cost_inr": None,
        "procurement_cost_display": "Government Procurement Price unavailable — institutional quote required.",
        "retail_price_inr": None,
        "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
        "price_type": "UNAVAILABLE",
        "farmer_cost_inr": None,
        "farmer_cost_display": RETAIL_UNAVAILABLE_MESSAGE,
        "cost_per_dose_display": RETAIL_UNAVAILABLE_MESSAGE,
        "state_market": "Specialized Tick-Endemic Dairy Pockets",
        "source_name": "Indian Immunologicals Ltd / State Veterinary Cold Chain Distribution",
        "source_url": "https://www.indimmune.com",
        "source_date": "2023-08-01",
        "recommended_timing": "Single dose in crossbred and exotic calves over 2 months of age.",
        "interval_days": None,
        "notes": "Requires continuous liquid nitrogen (-196°C) cold chain. Institutional quote required.",
        "eligibility_notes": "Requires liquid nitrogen cold chain (-196°C); check with State Livestock Development Board.",
    },
}


def calculate_per_dose_price(total_pack_price: Optional[float], pack_size_doses: Optional[int]) -> Optional[float]:
    """Calculates unit dose cost: total_pack_price / pack_size_doses."""
    if total_pack_price is not None and pack_size_doses is not None and pack_size_doses > 0:
        return round(float(total_pack_price) / float(pack_size_doses), 2)
    return None


def is_price_stale(source_date_str: Optional[str], threshold_days: int = 730) -> bool:
    """
    Determines if source date is older than configured threshold (default 2 years / 730 days).
    """
    if not source_date_str:
        return False
    try:
        if len(source_date_str) == 7:
            src_dt = datetime.strptime(source_date_str, "%Y-%m").date()
        else:
            src_dt = datetime.strptime(source_date_str[:10], "%Y-%m-%d").date()
        return (date.today() - src_dt).days > threshold_days
    except Exception:
        return False


class VaccinationService:
    """Production Veterinary Vaccination Scheduling & Authoritative Pricing Service."""

    def get_vaccination_schedule_config(self) -> Dict[str, Dict[str, Any]]:
        """Returns the active configured vaccination schedule and price benchmarks."""
        return STANDARD_VACCINATION_SCHEDULE

    def get_vaccine_price_detail(self, disease_code: str) -> VaccinePriceDetail:
        """
        Builds a structured VaccinePriceDetail object with per-dose calculations,
        procurement vs farmer pricing, source citation, price classification, and staleness evaluation.
        """
        code = disease_code.upper().strip()
        if code not in STANDARD_VACCINATION_SCHEDULE:
            return VaccinePriceDetail(
                disease_target=code,
                vaccine_name="Veterinary Vaccine (Specific to condition)",
                price_type="UNAVAILABLE",
                farmer_cost_inr=None,
                farmer_cost_display=RETAIL_UNAVAILABLE_MESSAGE,
                cost_per_dose_display=RETAIL_UNAVAILABLE_MESSAGE,
                procurement_cost_inr=None,
                procurement_cost_display="Government Procurement Price unavailable.",
                retail_price_inr=None,
                retail_price_display=RETAIL_UNAVAILABLE_MESSAGE,
                state_market="Local Veterinary Market",
                source_name="State Animal Husbandry Department / Local Veterinary Clinic",
                source_url=None,
                source_date=None,
                is_stale=False,
                notes="Consult local veterinarian for prescription and availability.",
                eligibility_notes=None
            )

        info = STANDARD_VACCINATION_SCHEDULE[code]
        calc_dose = info.get("calculated_per_dose_inr")
        if calc_dose is None and info.get("total_pack_price_inr") and info.get("pack_size_doses"):
            calc_dose = calculate_per_dose_price(info["total_pack_price_inr"], info["pack_size_doses"])

        stale = is_price_stale(info.get("source_date"))

        return VaccinePriceDetail(
            disease_target=code,
            vaccine_name=info["recommended_vaccine"],
            brand_name=info.get("brand_name"),
            manufacturer=info.get("manufacturer"),
            pack_size_doses=info.get("pack_size_doses"),
            total_pack_price_inr=info.get("total_pack_price_inr"),
            calculated_per_dose_inr=calc_dose,
            cost_per_dose_display=info["cost_per_dose_display"],
            procurement_cost_inr=info.get("procurement_cost_inr", calc_dose),
            procurement_cost_display=info.get("procurement_cost_display"),
            retail_price_inr=info.get("retail_price_inr"),
            retail_price_display=info.get("retail_price_display", RETAIL_UNAVAILABLE_MESSAGE),
            price_type=info["price_type"],
            farmer_cost_inr=info.get("farmer_cost_inr"),
            farmer_cost_display=info["farmer_cost_display"],
            state_market=info.get("state_market", "All India"),
            source_name=info.get("source_name"),
            source_url=info.get("source_url"),
            source_date=info.get("source_date"),
            is_stale=stale,
            notes=info.get("notes"),
            eligibility_notes=info.get("eligibility_notes")
        )

    def generate_recommendations(
        self,
        cattle: Cattle,
        administered_records: Optional[List[VaccinationRecord]] = None
    ) -> List[VaccinationRecommendation]:
        """
        Computes tailored vaccination recommendations for an animal based on
        its age, species, breed, and past vaccination history with source-backed pricing.
        """
        administered_map: Dict[str, VaccinationRecord] = {}
        if administered_records:
            for r in administered_records:
                target = r.disease_target.upper().strip()
                if target not in administered_map or r.administered_date > administered_map[target].administered_date:
                    administered_map[target] = r

        today_dt = date.today()
        recommendations: List[VaccinationRecommendation] = []

        for disease_key, sched in STANDARD_VACCINATION_SCHEDULE.items():
            if disease_key == "BRUCELLOSIS" and cattle.gender.lower() == "male":
                continue

            last_record = administered_map.get(disease_key.upper())
            interval = sched.get("interval_days")
            last_admin_date_str = None

            if last_record:
                last_admin_date_str = last_record.administered_date
                try:
                    admin_date = datetime.strptime(last_record.administered_date, "%Y-%m-%d").date()
                except ValueError:
                    admin_date = today_dt

                if interval:
                    next_due = admin_date + timedelta(days=interval)
                    days_remaining = (next_due - today_dt).days

                    if days_remaining < 0:
                        status = "OVERDUE"
                    elif days_remaining <= 14:
                        status = "DUE"
                    else:
                        status = "UPCOMING"
                    next_due_str = next_due.isoformat()
                else:
                    # Lifetime single dose
                    status = "COMPLETED"
                    next_due_str = "LIFETIME_PROTECTED"
            else:
                # Never recorded -> Due now or check eligibility
                age_mo = cattle.age_months or 24.0
                if disease_key == "BRUCELLOSIS" and age_mo > 12.0:
                    status = "NOT_APPLICABLE_ADULT"
                    next_due_str = "N/A (Exceeded 8 mo calf window)"
                else:
                    status = "DUE"
                    next_due_str = today_dt.isoformat()

            price_detail = self.get_vaccine_price_detail(disease_key)

            recommendations.append(
                VaccinationRecommendation(
                    tag_id=cattle.tag_id,
                    disease_target=disease_key,
                    recommended_vaccine=sched["recommended_vaccine"],
                    recommended_timing=sched["recommended_timing"],
                    next_due_date=next_due_str,
                    status=status,
                    estimated_cost_inr=price_detail.calculated_per_dose_inr or 0.0 if price_detail.price_type != "UNAVAILABLE" else None,
                    estimated_cost_display=price_detail.farmer_cost_display,
                    brand_name=price_detail.brand_name,
                    manufacturer=price_detail.manufacturer,
                    pack_size_doses=price_detail.pack_size_doses,
                    total_pack_price_inr=price_detail.total_pack_price_inr,
                    calculated_per_dose_inr=price_detail.calculated_per_dose_inr,
                    procurement_cost_inr=price_detail.procurement_cost_inr,
                    procurement_cost_display=price_detail.procurement_cost_display,
                    retail_price_inr=price_detail.retail_price_inr,
                    retail_price_display=price_detail.retail_price_display,
                    price_type=price_detail.price_type,
                    farmer_cost_inr=price_detail.farmer_cost_inr,
                    farmer_cost_display=price_detail.farmer_cost_display,
                    state_market=price_detail.state_market,
                    source_name=price_detail.source_name,
                    source_url=price_detail.source_url,
                    source_date=price_detail.source_date,
                    is_stale=price_detail.is_stale,
                    eligibility_notes=price_detail.eligibility_notes,
                    price_detail=price_detail,
                    last_administered_date=last_admin_date_str,
                    notes=sched["notes"]
                )
            )

        return recommendations

    def get_vaccine_info_for_disease(self, disease_code: str) -> Dict[str, Any]:
        """
        Retrieves source-backed vaccine, timing, and pricing details for a diagnosed disease condition.
        """
        code = disease_code.upper().strip()
        price_detail = self.get_vaccine_price_detail(code)

        if code in STANDARD_VACCINATION_SCHEDULE:
            info = STANDARD_VACCINATION_SCHEDULE[code]
            return {
                "recommended_vaccine": info["recommended_vaccine"],
                "vaccination_timing": info["recommended_timing"],
                "estimated_cost": price_detail.farmer_cost_display,
                "brand_name": price_detail.brand_name,
                "manufacturer": price_detail.manufacturer,
                "price_type": price_detail.price_type,
                "farmer_cost_display": price_detail.farmer_cost_display,
                "calculated_per_dose_inr": price_detail.calculated_per_dose_inr,
                "procurement_cost_display": price_detail.procurement_cost_display,
                "retail_price_display": price_detail.retail_price_display,
                "source_name": price_detail.source_name,
                "source_url": price_detail.source_url,
                "source_date": price_detail.source_date,
                "is_stale": price_detail.is_stale,
                "eligibility_notes": price_detail.eligibility_notes,
                "price_detail": price_detail,
                "veterinary_disclaimer": "Estimated information only. Consult a qualified veterinarian for diagnosis and vaccination decisions."
            }
        elif code == "IBK":
            return {
                "recommended_vaccine": "Moraxella bovis bacterin / Autogenous pinkeye vaccine (where available)",
                "vaccination_timing": "Administer 4-6 weeks prior to peak fly season; topical antibiotic eye therapy recommended for acute cases.",
                "estimated_cost": RETAIL_UNAVAILABLE_MESSAGE,
                "brand_name": "Autogenous Bacterin",
                "manufacturer": "Specialized Veterinary Formulations",
                "price_type": "UNAVAILABLE",
                "farmer_cost_display": RETAIL_UNAVAILABLE_MESSAGE,
                "calculated_per_dose_inr": None,
                "procurement_cost_display": "Government Procurement Price unavailable.",
                "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
                "source_name": "State Veterinary College & Research Institute / Local Dispensary",
                "source_url": None,
                "source_date": None,
                "is_stale": False,
                "eligibility_notes": "Commercial vaccines have limited availability in retail market; topical ocular therapy is standard field protocol.",
                "price_detail": price_detail,
                "veterinary_disclaimer": "Estimated information only. Consult a qualified veterinarian for diagnosis and vaccination decisions."
            }
        elif code == "NORMAL":
            return {
                "recommended_vaccine": "Maintain routine herd vaccination protocol (FMD, HS, BQ, LSD)",
                "vaccination_timing": "According to regular state veterinary department calendar",
                "estimated_cost": "Routine vaccinations provided free or subsidized through State AH Dept",
                "brand_name": "Government Routine Herd Protocols",
                "manufacturer": "DAHD / State Animal Husbandry",
                "price_type": "GOVERNMENT_PROGRAMME_FREE",
                "farmer_cost_display": "₹0 under routine government vaccination drives",
                "calculated_per_dose_inr": 0.0,
                "procurement_cost_display": "Procurement varies by state health schedule.",
                "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
                "source_name": "Department of Animal Husbandry & Dairying (DAHD)",
                "source_url": "https://dahd.nic.in",
                "source_date": "2024-01-15",
                "is_stale": False,
                "eligibility_notes": "All dairy animals eligible under NADCP and state health schedules.",
                "price_detail": price_detail,
                "veterinary_disclaimer": "Estimated information only. Consult a qualified veterinarian for herd health planning."
            }
        else:
            return {
                "recommended_vaccine": "Consult a veterinarian for the appropriate vaccine.",
                "vaccination_timing": "As prescribed by a registered veterinary practitioner.",
                "estimated_cost": RETAIL_UNAVAILABLE_MESSAGE,
                "brand_name": None,
                "manufacturer": None,
                "price_type": "UNAVAILABLE",
                "farmer_cost_display": RETAIL_UNAVAILABLE_MESSAGE,
                "calculated_per_dose_inr": None,
                "procurement_cost_display": "Government Procurement Price unavailable.",
                "retail_price_display": RETAIL_UNAVAILABLE_MESSAGE,
                "source_name": "Local Veterinary Clinic / Animal Husbandry Department",
                "source_url": None,
                "source_date": None,
                "is_stale": False,
                "eligibility_notes": None,
                "price_detail": price_detail,
                "veterinary_disclaimer": "Estimated information only. Consult a qualified veterinarian for diagnosis and vaccination decisions."
            }


vaccination_service = VaccinationService()
