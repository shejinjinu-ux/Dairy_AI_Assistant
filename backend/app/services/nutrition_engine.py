"""
Deterministic Field Nutrition & Least-Cost Ration Optimization Engine
Based strictly on ICAR-2013/2024 Standards and ICAR-NIANP Indian Feed Composition Database
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import scipy.optimize

from backend.app.schemas.nutrition import (
    NutritionRecommendationRequest,
    NutritionRecommendationResponse,
    NutrientRequirementsSummary,
    FeedItemRecommendation,
    NutrientBalanceItem
)

logger = logging.getLogger("dairy_ai.nutrition_engine")

# Path to verified ICAR-NIANP feed composition dataset
FEED_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "icar_nianp_indian_feed_composition.csv")
)

# Standard Indicative Market Prices in India (INR per kg fresh feed)
DEFAULT_FEED_PRICES: Dict[str, float] = {
    # Green Roughages
    "IN_GF_001": 2.50,  # Maize Fodder (Green)
    "IN_GF_002": 2.20,  # Sorghum / Jowar Fodder (Green)
    "IN_GF_003": 2.00,  # Bajra / Pearl Millet Fodder (Green)
    "IN_GF_004": 2.50,  # Hybrid Napier (CO-3 / CO-4 / CO-5)
    "IN_GF_005": 2.20,  # Guinea Grass
    "IN_GF_006": 2.00,  # Para Grass
    "IN_GF_007": 3.00,  # Oat Fodder
    "IN_GL_008": 3.50,  # Berseem (Green)
    "IN_GL_009": 4.00,  # Lucerne / Alfalfa (Green)
    "IN_GL_010": 3.20,  # Cowpea Fodder
    "IN_GL_011": 3.00,  # Hedge Lucerne
    "IN_GL_012": 1.50,  # Azolla pinnata

    # Dry Roughages
    "IN_DR_013": 7.00,  # Paddy Straw
    "IN_DR_014": 8.00,  # Wheat Straw
    "IN_DR_015": 7.50,  # Sorghum Stover / Kadbi
    "IN_DR_016": 6.50,  # Maize Stover
    "IN_DR_017": 5.00,  # Sugarcane Tops

    # Silages
    "IN_SI_018": 4.50,  # Maize Silage
    "IN_SI_019": 4.00,  # Sorghum Silage

    # Concentrates / Grains
    "IN_EC_020": 24.00, # Maize Grain
    "IN_EC_021": 22.00, # Barley Grain
    "IN_EC_022": 26.00, # Wheat Grain

    # Protein Meals / Cakes
    "IN_PC_023": 32.00, # Cottonseed Cake (Decorticated)
    "IN_PC_024": 26.00, # Cottonseed Cake (Undecorticated)
    "IN_PC_025": 28.00, # Mustard / Rapeseed Cake
    "IN_PC_026": 38.00, # Groundnut Cake
    "IN_PC_027": 36.00, # Soybean Meal
    "IN_PC_028": 30.00, # Sesame / Til Cake

    # Byproducts
    "IN_BP_029": 22.00, # Wheat Bran (Chokar)
    "IN_BP_030": 16.00, # De-Oiled Rice Bran (DORB)
    "IN_BP_031": 20.00, # Rice Polish
    "IN_BP_032": 24.00, # Gram / Chickpea Chuni
    "IN_BP_033": 23.00, # Tur / Arhar Chuni
    "IN_BP_034": 12.00, # Sugarcane Molasses

    # Compound Feed & Mineral Supplements
    "IN_CF_035": 24.00, # Compound Cattle Feed (BIS Type-II)
    "IN_CF_036": 28.00, # Compound Cattle Feed (BIS Type-I High Yield)
    "IN_CF_037": 80.00, # Area Specific Mineral Mixture (ASMM)
    "IN_CF_038": 150.00 # Bypass Fat
}

# Standard Default Feed Portfolio for Indian Smallholders when unconstrained
DEFAULT_FEED_PORTFOLIO_IDS: List[str] = [
    "IN_GF_004", # Hybrid Napier (Green)
    "IN_DR_013", # Paddy Straw (Dry)
    "IN_PC_023", # Cottonseed Cake (Decorticated)
    "IN_CF_035", # Compound Cattle Feed (BIS Type-II)
    "IN_CF_037"  # Area Specific Mineral Mixture (ASMM)
]


class ICARFieldNutritionEngine:
    """
    Scientific Deterministic Ration Balancing Engine.
    Executes ICAR requirement calculations and Linear Programming least-cost optimization.
    """

    def __init__(self, feed_dataset_path: str = FEED_DATASET_PATH):
        self.feed_dataset_path = feed_dataset_path
        self._feed_db: pd.DataFrame = self._load_feed_database()

    def _load_feed_database(self) -> pd.DataFrame:
        """Loads and indexes the verified ICAR-NIANP Feed Composition Database."""
        try:
            if os.path.exists(self.feed_dataset_path):
                df = pd.read_csv(self.feed_dataset_path)
                logger.info(f"Loaded {len(df)} feed ingredients from ICAR-NIANP database.")
                return df
            else:
                logger.warning(f"Feed dataset not found at {self.feed_dataset_path}. Using empty DB.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading ICAR-NIANP feed dataset: {e}")
            return pd.DataFrame()

    def get_all_feeds(self) -> List[Dict[str, Any]]:
        """Returns all verified feed ingredients with nutrient profiles and default prices."""
        if self._feed_db.empty:
            return []
        records = []
        for _, row in self._feed_db.iterrows():
            f_dict = row.to_dict()
            f_dict["default_cost_per_kg_inr"] = DEFAULT_FEED_PRICES.get(f_dict["feed_id"], 20.0)
            records.append(f_dict)
        return records

    def calculate_icar_requirements(
        self,
        species: str,
        breed_type: str,
        body_weight_kg: float,
        daily_milk_yield_kg: float,
        milk_fat_percent: float,
        pregnancy_month: Optional[int] = None
    ) -> NutrientRequirementsSummary:
        """
        Calculates exact nutrient requirements using ICAR-2013 / 2024 published partition formulas.
        """
        # 1. Metabolic Body Weight
        mbw = body_weight_kg ** 0.75

        # 2. Maintenance Requirements
        is_buffalo = "buffalo" in species.lower()
        is_crossbred = "cross" in breed_type.lower() or "hf" in breed_type.lower() or "jersey" in breed_type.lower()

        if is_buffalo:
            maint_dmi = 0.023 * body_weight_kg
            maint_tdn = 0.035 * mbw
            maint_cp = 4.3 * mbw
            maint_ca = 0.052 * body_weight_kg
            maint_p = 0.036 * body_weight_kg
            lact_tdn_coeff = 0.340
            lact_cp_coeff = 92.0
            lact_ca_coeff = 3.5
            lact_p_coeff = 2.4
            lact_dmi_coeff = 0.36
        elif is_crossbred:
            maint_dmi = 0.024 * body_weight_kg
            maint_tdn = 0.036 * mbw
            maint_cp = 4.5 * mbw
            maint_ca = 0.055 * body_weight_kg
            maint_p = 0.038 * body_weight_kg
            lact_tdn_coeff = 0.325
            lact_cp_coeff = 88.0
            lact_ca_coeff = 3.2
            lact_p_coeff = 2.1
            lact_dmi_coeff = 0.35
        else: # Indigenous Zebu Cattle (Gir, Sahiwal, Kankrej, Tharparkar, Red Sindhi, etc.)
            maint_dmi = 0.022 * body_weight_kg
            maint_tdn = 0.034 * mbw
            maint_cp = 4.2 * mbw
            maint_ca = 0.050 * body_weight_kg
            maint_p = 0.035 * body_weight_kg
            lact_tdn_coeff = 0.320
            lact_cp_coeff = 85.0
            lact_ca_coeff = 3.0
            lact_p_coeff = 2.0
            lact_dmi_coeff = 0.33

        # 3. Lactation Requirements (based on 4% Fat-Corrected Milk)
        fcm_4pct = (0.4 + 0.15 * milk_fat_percent) * daily_milk_yield_kg
        lact_tdn = lact_tdn_coeff * fcm_4pct
        lact_cp = lact_cp_coeff * fcm_4pct
        lact_ca = lact_ca_coeff * fcm_4pct
        lact_p = lact_p_coeff * fcm_4pct
        lact_dmi = lact_dmi_coeff * daily_milk_yield_kg

        # 4. Pregnancy Allowance (last trimester: month 7, 8, 9)
        preg_dmi = 0.0
        preg_tdn = 0.0
        preg_cp = 0.0
        preg_ca = 0.0
        preg_p = 0.0
        if pregnancy_month and pregnancy_month >= 7:
            preg_dmi = 1.0
            preg_tdn = 1.20
            preg_cp = 250.0
            preg_ca = 12.0
            preg_p = 8.0

        # 5. Sum Totals
        total_dmi = round(maint_dmi + lact_dmi + preg_dmi, 2)
        total_tdn = round(maint_tdn + lact_tdn + preg_tdn, 2)
        total_me = round(total_tdn * 15.1, 1) # 1 kg TDN ≈ 15.1 MJ Metabolizable Energy
        total_cp = round(maint_cp + lact_cp + preg_cp, 1)
        total_ca = round(maint_ca + lact_ca + preg_ca, 1)
        total_p = round(maint_p + lact_p + preg_p, 1)

        return NutrientRequirementsSummary(
            metabolic_body_weight_kg=round(mbw, 2),
            fat_corrected_milk_4pct_kg=round(fcm_4pct, 2),
            req_dmi_kg_per_day=total_dmi,
            req_tdn_kg_per_day=total_tdn,
            req_me_mj_per_day=total_me,
            req_cp_g_per_day=total_cp,
            req_calcium_g_per_day=total_ca,
            req_phosphorus_g_per_day=total_p
        )

    def _resolve_feed_selection(
        self,
        available_feeds: Optional[List[str]],
        feed_prices: Optional[Dict[str, float]]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Filters and selects feed ingredients for optimization, establishing prices.
        """
        if self._feed_db.empty:
            return pd.DataFrame(), np.array([])

        selected_rows = []
        if available_feeds and len(available_feeds) > 0:
            # Match available feeds by name, ID, or keywords
            for _, row in self._feed_db.iterrows():
                f_name = row["feed_name"].lower()
                f_id = row["feed_id"].lower()
                f_cat = row["feed_category"].lower()
                
                matched = False
                for af in available_feeds:
                    af_clean = af.strip().lower()
                    if (af_clean in f_name or af_clean == f_id or 
                        f_name in af_clean or af_clean in f_cat):
                        matched = True
                        break
                if matched:
                    selected_rows.append(row)

        # Fallback to default diverse portfolio if selection is empty or too narrow
        if len(selected_rows) < 3:
            selected_rows = [
                row for _, row in self._feed_db.iterrows()
                if row["feed_id"] in DEFAULT_FEED_PORTFOLIO_IDS
            ]

        selected_df = pd.DataFrame(selected_rows).drop_duplicates(subset=["feed_id"]).reset_index(drop=True)

        # Build cost vector
        costs = []
        for _, row in selected_df.iterrows():
            f_id = row["feed_id"]
            f_name = row["feed_name"]
            
            # Check user custom price override
            cost = None
            if feed_prices:
                if f_id in feed_prices:
                    cost = feed_prices[f_id]
                elif f_name in feed_prices:
                    cost = feed_prices[f_name]
                else:
                    # Keyword check
                    for pk, pv in feed_prices.items():
                        if pk.lower() in f_name.lower():
                            cost = pv
                            break
            if cost is None:
                cost = DEFAULT_FEED_PRICES.get(f_id, 20.0)
            costs.append(cost)

        return selected_df, np.array(costs)

    def optimize_ration(
        self,
        request: NutritionRecommendationRequest
    ) -> NutritionRecommendationResponse:
        """
        Executes deterministic Least-Cost Ration Formulation using Linear Programming.
        """
        # 1. Input Validation
        missing = []
        if request.body_weight_kg is None or request.body_weight_kg <= 0:
            missing.append("body_weight_kg")
        if request.daily_milk_yield_kg is None or request.daily_milk_yield_kg < 0:
            missing.append("daily_milk_yield_kg")
        elif request.daily_milk_yield_kg > 0 and (request.milk_fat_percent is None or request.milk_fat_percent <= 0):
            missing.append("milk_fat_percent")

        if missing:
            return NutritionRecommendationResponse(
                success=False,
                is_deterministic_optimized=True,
                status="missing_parameters",
                message=f"Missing critical parameters: {', '.join(missing)}. Please provide body weight, daily milk yield, and milk fat %.",
                animal_profile=request.model_dump(exclude_none=True),
                missing_critical_parameters=missing
            )

        # Biological ranges sanity check
        bw = float(request.body_weight_kg)
        my = float(request.daily_milk_yield_kg)
        fat = float(request.milk_fat_percent) if (my > 0 and request.milk_fat_percent is not None) else 0.0
        breed_str = request.breed or "Indigenous_Zebu"


        if bw < 100 or bw > 1200:
            return NutritionRecommendationResponse(
                success=False,
                status="invalid_parameters",
                message=f"Provided body weight ({bw} kg) is outside biological limits for Indian bovines (150 - 900 kg).",
                missing_critical_parameters=["valid_body_weight"]
            )
        if my > 70:
            return NutritionRecommendationResponse(
                success=False,
                status="invalid_parameters",
                message=f"Provided daily milk yield ({my} kg) is outside biological limits for Indian bovines.",
                missing_critical_parameters=["valid_milk_yield"]
            )

        # 2. Compute ICAR Requirements
        req = self.calculate_icar_requirements(
            species=request.species,
            breed_type=breed_str,
            body_weight_kg=bw,
            daily_milk_yield_kg=my,
            milk_fat_percent=fat,
            pregnancy_month=request.pregnancy_month if request.pregnancy_status else None
        )

        # 3. Select Feeds and Formulate Linear Program
        selected_df, c_costs = self._resolve_feed_selection(
            available_feeds=request.available_feeds,
            feed_prices=request.feed_prices
        )

        if selected_df.empty:
            return NutritionRecommendationResponse(
                success=False,
                status="infeasible",
                message="Feed database is unavailable or no valid feeds could be selected.",
                nutrient_requirements=req
            )

        n_feeds = len(selected_df)

        # Nutrient vectors (fresh basis)
        dm = selected_df["dry_matter_percent"].values / 100.0
        tdn = (selected_df["tdn_dm_pct"].values / 100.0) * dm
        cp = (selected_df["crude_protein_dm_pct"].values / 100.0) * dm * 1000.0 # g CP / kg fresh
        ca = selected_df["calcium_g_per_kg"].values * dm # g Ca / kg fresh
        p = selected_df["phosphorus_g_per_kg"].values * dm # g P / kg fresh

        # Inequality Constraints: A_ub * x <= b_ub
        # 1. Total DM <= req.req_dmi_kg_per_day * 1.05
        # 2. Total DM >= req.req_dmi_kg_per_day * 0.85
        # 3. Total TDN >= req.req_tdn_kg_per_day
        # 4. Total CP >= req.req_cp_g_per_day
        # 5. Total Ca >= req.req_calcium_g_per_day
        # 6. Total P >= req.req_phosphorus_g_per_day

        A_ub = np.array([
            dm,
            -dm,
            -tdn,
            -cp,
            -ca,
            -p
        ])
        b_ub = np.array([
            req.req_dmi_kg_per_day * 1.05,
            -req.req_dmi_kg_per_day * 0.85,
            -req.req_tdn_kg_per_day,
            -req.req_cp_g_per_day,
            -req.req_calcium_g_per_day,
            -req.req_phosphorus_g_per_day
        ])

        # Bounds per feed item (ensuring rumen health and palatability)
        bounds = []
        for _, row in selected_df.iterrows():
            cat = row["feed_category"]
            subcat = row.get("feed_subcategory", "")
            f_id = row["feed_id"]

            if "Mineral" in cat or f_id == "IN_CF_037":
                # Mineral mixture: 50g - 100g (0.05 - 0.10 kg)
                bounds.append((0.05, 0.12))
            elif "Green" in cat:
                # Green fodder: minimum 5kg up to 35kg
                bounds.append((5.0, 35.0))
            elif "Dry" in cat:
                # Dry roughage: minimum 2.5kg up to 9kg (essential for cud-chewing)
                bounds.append((2.5, 9.0))
            elif "Silage" in cat:
                # Silage: 0 to 15kg
                bounds.append((0.0, 15.0))
            elif "Energy Supplement" in cat or f_id == "IN_CF_038":
                # Bypass fat: 0 to 0.3kg
                bounds.append((0.0, 0.30))
            else: # Concentrate, Cake, Byproduct, Compound Feed
                bounds.append((0.0, 8.0))

        # 4. Solve Linear Program
        res = scipy.optimize.linprog(
            c=c_costs,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs"
        )

        warnings = []
        if not res.success:
            # If infeasible with selected feeds, loosen bounds or report infeasibility
            logger.warning(f"Linear program failed with status: {res.message}. Attempting generalized solver fallback.")
            # Relaxed bounds
            relaxed_bounds = [(0.0, 40.0) if "Green" in r["feed_category"] else (0.0, 15.0) for _, r in selected_df.iterrows()]
            res = scipy.optimize.linprog(c=c_costs, A_ub=A_ub, b_ub=b_ub, bounds=relaxed_bounds, method="highs")
            if not res.success:
                return NutritionRecommendationResponse(
                    success=False,
                    status="infeasible",
                    message=(
                        "Ration optimization could not find a feasible balanced diet with the specified feeds. "
                        "Please add a green fodder, protein cake (cottonseed/mustard), and mineral mixture."
                    ),
                    animal_profile=request.model_dump(exclude_none=True),
                    nutrient_requirements=req,
                    warnings=["Nutrient demands exceed capacity of available feed ingredients."]
                )
            warnings.append("Relaxed feeding constraints were applied to reach nutritional feasibility.")

        # 5. Extract Recommended Ration and Balances
        recommended_ration: List[FeedItemRecommendation] = []
        total_daily_cost = 0.0

        for i, row in selected_df.iterrows():
            qty = round(float(res.x[i]), 2)
            if qty > 0.01: # Include items with non-trivial quantity
                c_unit = float(c_costs[i])
                item_cost = round(qty * c_unit, 2)
                total_daily_cost += item_cost
                
                dm_item = round(qty * dm[i], 2)
                cp_item = round(qty * cp[i], 1)
                tdn_item = round(qty * tdn[i], 2)
                ca_item = round(qty * ca[i], 1)
                p_item = round(qty * p[i], 1)

                recommended_ration.append(FeedItemRecommendation(
                    feed_id=row["feed_id"],
                    feed_name=row["feed_name"],
                    feed_category=row["feed_category"],
                    quantity_kg_per_day=qty,
                    cost_per_kg_inr=c_unit,
                    daily_cost_inr=item_cost,
                    dm_supplied_kg=dm_item,
                    cp_supplied_g=cp_item,
                    tdn_supplied_kg=tdn_item,
                    calcium_supplied_g=ca_item,
                    phosphorus_supplied_g=p_item
                ))

        # Supplied totals
        supplied_dm = round(float(np.dot(dm, res.x)), 2)
        supplied_tdn = round(float(np.dot(tdn, res.x)), 2)
        supplied_me = round(supplied_tdn * 15.1, 1)
        supplied_cp = round(float(np.dot(cp, res.x)), 1)
        supplied_ca = round(float(np.dot(ca, res.x)), 1)
        supplied_p = round(float(np.dot(p, res.x)), 1)

        total_daily_cost = round(total_daily_cost, 2)

        nutrient_supply = {
            "dry_matter_kg": supplied_dm,
            "tdn_kg": supplied_tdn,
            "me_mj": supplied_me,
            "crude_protein_g": supplied_cp,
            "calcium_g": supplied_ca,
            "phosphorus_g": supplied_p
        }

        # 6. Nutrient Balance Table
        def make_balance_item(req_val: float, sup_val: float, unit: str) -> NutrientBalanceItem:
            diff = round(sup_val - req_val, 2)
            pct = round((sup_val / req_val * 100.0), 1) if req_val > 0 else 100.0
            if pct >= 98.0 and pct <= 115.0:
                stat = "Balanced"
            elif pct > 115.0:
                stat = "Surplus"
            else:
                stat = "Deficit"
            return NutrientBalanceItem(
                required=req_val,
                supplied=sup_val,
                unit=unit,
                difference=diff,
                percentage_fulfilled=pct,
                status=stat
            )

        nutrient_balance = {
            "dry_matter": make_balance_item(req.req_dmi_kg_per_day, supplied_dm, "kg/day"),
            "tdn": make_balance_item(req.req_tdn_kg_per_day, supplied_tdn, "kg/day"),
            "crude_protein": make_balance_item(req.req_cp_g_per_day, supplied_cp, "g/day"),
            "calcium": make_balance_item(req.req_calcium_g_per_day, supplied_ca, "g/day"),
            "phosphorus": make_balance_item(req.req_phosphorus_g_per_day, supplied_p, "g/day")
        }

        # Quality warnings
        if supplied_dm > req.req_dmi_kg_per_day * 1.05:
            warnings.append("Total Dry Matter slightly exceeds standard capacity; ensure clean potable water access.")
        if supplied_cp < req.req_cp_g_per_day:
            warnings.append("Crude Protein is borderline deficient; consider adding 0.5kg oil cake.")

        return NutritionRecommendationResponse(
            success=True,
            is_deterministic_optimized=True,
            status="optimized",
            message=f"Optimal least-cost balanced ration calculated successfully for {bw} kg {request.species} yielding {my} kg milk/day.",
            animal_profile=request.model_dump(exclude_none=True),
            missing_critical_parameters=[],
            nutrient_requirements=req,
            recommended_ration=recommended_ration,
            total_daily_cost_inr=total_daily_cost,
            nutrient_supply=nutrient_supply,
            nutrient_balance=nutrient_balance,
            warnings=warnings
        )


# Global singleton optimization engine
nutrition_engine = ICARFieldNutritionEngine()
