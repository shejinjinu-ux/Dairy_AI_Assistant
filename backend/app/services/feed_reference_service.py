"""
Feed Composition Reference Service
Provides deterministic nutrition calculations grounded in ICAR-NIANP, Feedipedia, and BIS standards.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings
from backend.app.schemas.feed_reference import (
    FeedReferenceRequest,
    FeedReferenceResponse,
    FeedReferenceItem,
    FeedCatalogResponse,
    NutrientProfile
)

logger = logging.getLogger("dairy_ai.feed_reference")


class FeedReferenceService:
    """Service for authoritative feed composition reference lookups and calculations."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or (settings.PROJECT_ROOT / "data" / "feed_composition.csv")
        self._catalog: Dict[str, Dict[str, Any]] = {}
        self._synonym_map: Dict[str, str] = {}
        self._load_database()

    def _load_database(self) -> None:
        """Load and parse the feed reference CSV database."""
        if not self.csv_path.exists():
            logger.error(f"Feed reference database CSV not found at: {self.csv_path}")
            return

        try:
            with open(self.csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    feed_name = row["feed_name"].strip()
                    if not feed_name:
                        continue

                    item = {
                        "feed_name": feed_name,
                        "category": row.get("category", "Roughage").strip(),
                        "dry_matter_pct": float(row["dry_matter_pct"]),
                        "crude_protein_pct": float(row["crude_protein_pct"]),
                        "crude_fibre_pct": float(row["crude_fibre_pct"]),
                        "ndf_pct": float(row["ndf_pct"]),
                        "adf_pct": float(row["adf_pct"]),
                        "adl_pct": float(row["adl_pct"]) if row.get("adl_pct") and row["adl_pct"].strip() != "" and row["adl_pct"].lower() != "null" else None,
                        "starch_pct": float(row["starch_pct"]) if row.get("starch_pct") and row["starch_pct"].strip() != "" and row["starch_pct"].lower() != "null" else None,
                        "ether_extract_pct": float(row["ether_extract_pct"]),
                        "ash_pct": float(row["ash_pct"]),
                        "energy_mj_kg": float(row["energy_mj_kg"]),
                        "calcium_g_kg": float(row["calcium_g_kg"]),
                        "phosphorus_g_kg": float(row["phosphorus_g_kg"]),
                        "source": row.get("source", "ICAR-NIANP Indian Feed Composition Tables (2013/2024)").strip()
                    }
                    self._catalog[feed_name.lower()] = item

            self._build_synonym_map()
            logger.info(f"Loaded {len(self._catalog)} verified feed reference entries from {self.csv_path.name}")
        except Exception as e:
            logger.error(f"Failed to load feed reference database: {e}")

    def _build_synonym_map(self) -> None:
        """Create mapping from common colloquial / regional terms to canonical feed names."""
        synonyms = {
            # Maize
            "maize": "Maize Grain",
            "corn": "Maize Grain",
            "makka": "Maize Grain",
            "makki": "Maize Grain",
            "maize grain": "Maize Grain",
            "corn grain": "Maize Grain",
            "maize fodder": "Green Fodder",
            "maize green": "Green Fodder",
            "green maize": "Green Fodder",
            "corn silage": "Maize Silage",
            "maize silage": "Maize Silage",
            "silage": "Maize Silage",
            # Grasses / Green Roughage
            "napier": "Hybrid Napier",
            "napier grass": "Hybrid Napier",
            "hybrid napier": "Hybrid Napier",
            "cumbu napier": "Hybrid Napier",
            "co-3": "Hybrid Napier",
            "co-4": "Hybrid Napier",
            "co-5": "Hybrid Napier",
            "guinea grass": "Guinea Grass",
            "para grass": "Para Grass",
            "oat": "Oat Fodder",
            "oat fodder": "Oat Fodder",
            "berseem": "Berseem",
            "egyptian clover": "Berseem",
            "lucerne": "Lucerne",
            "alfalfa": "Lucerne",
            "cowpea": "Cowpea Fodder",
            "cowpea fodder": "Cowpea Fodder",
            "hedge lucerne": "Hedge Lucerne",
            "desmanthus": "Hedge Lucerne",
            "velvet bean": "Cowpea Fodder",
            "azolla": "Azolla",
            "green fodder": "Hybrid Napier",
            "green grass": "Hybrid Napier",
            # Sorghum / Jowar
            "sorghum": "Sorghum Fodder",
            "jowar": "Sorghum Fodder",
            "sorghum fodder": "Sorghum Fodder",
            "jowar fodder": "Sorghum Fodder",
            "sorghum silage": "Sorghum Silage",
            "jowar silage": "Sorghum Silage",
            "sorghum stover": "Sorghum Stover",
            "kadbi": "Sorghum Stover",
            "jowar kadbi": "Sorghum Stover",
            # Bajra
            "bajra": "Bajra Fodder",
            "pearl millet": "Bajra Fodder",
            "bajra fodder": "Bajra Fodder",
            # Ragi
            "ragi": "Ragi Grain",
            "finger millet": "Ragi Grain",
            "ragi grain": "Ragi Grain",
            "ragi straw": "Ragi Straw",
            "ragi stover": "Ragi Straw",
            # Straws / Dry Roughage
            "paddy straw": "Paddy Straw",
            "rice straw": "Paddy Straw",
            "parali": "Paddy Straw",
            "hay": "Paddy Straw",
            "dry fodder": "Paddy Straw",
            "dry roughage": "Paddy Straw",
            "wheat straw": "Wheat Straw",
            "bhusa": "Wheat Straw",
            "turi": "Wheat Straw",
            "maize stover": "Maize Stover",
            "sugarcane tops": "Sugarcane Tops",
            # Cakes & Meals
            "cottonseed cake": "Cottonseed Cake Decorticated",
            "cotton seed cake": "Cottonseed Cake Decorticated",
            "binola": "Cottonseed Cake Decorticated",
            "binola khal": "Cottonseed Cake Decorticated",
            "undecorticated cottonseed cake": "Cottonseed Cake Undecorticated",
            "mustard cake": "Mustard Cake",
            "sarson khal": "Mustard Cake",
            "rapeseed cake": "Mustard Cake",
            "groundnut cake": "Groundnut Cake",
            "peanut cake": "Groundnut Cake",
            "mungfali khal": "Groundnut Cake",
            "soybean meal": "Soybean Meal",
            "soya meal": "Soybean Meal",
            "doc soya": "Soybean Meal",
            "sesame cake": "Sesame Cake",
            "til cake": "Sesame Cake",
            # Brans & Chunis
            "wheat bran": "Wheat Bran",
            "chokar": "Wheat Bran",
            "rice bran": "Raw Rice Bran",
            "raw rice bran": "Raw Rice Bran",
            "dorb": "De-Oiled Rice Bran",
            "de-oiled rice bran": "De-Oiled Rice Bran",
            "deoiled rice bran": "De-Oiled Rice Bran",
            "gram chuni": "Gram Chuni",
            "chana chuni": "Gram Chuni",
            "tur chuni": "Tur Chuni",
            "arhar chuni": "Tur Chuni",
            "molasses": "Sugarcane Molasses",
            "sugarcane molasses": "Sugarcane Molasses",
            # Compound Feeds & Supplements
            "concentrate feed": "Concentrate Feed Type II",
            "compound cattle feed": "Concentrate Feed Type II",
            "cattle feed": "Concentrate Feed Type II",
            "concentrate": "Concentrate Feed Type II",
            "dairy concentrate": "Concentrate Feed Type I",
            "type 1 feed": "Concentrate Feed Type I",
            "type 2 feed": "Concentrate Feed Type II",
            "mineral mixture": "Area Specific Mineral Mixture",
            "asmm": "Area Specific Mineral Mixture",
            "minerals": "Area Specific Mineral Mixture",
            "bypass fat": "Bypass Fat",
            "rumen protected fat": "Bypass Fat",
        }
        for syn, target in synonyms.items():
            self._synonym_map[syn.lower().strip()] = target

    def find_feed(self, query_name: str) -> Optional[Dict[str, Any]]:
        """Resolve feed query by exact name, canonical synonym, or fuzzy substring."""
        cleaned = query_name.lower().strip()
        if not cleaned:
            return None

        # 1. Exact catalog match
        if cleaned in self._catalog:
            return self._catalog[cleaned]

        # 2. Synonym match
        if cleaned in self._synonym_map:
            target = self._synonym_map[cleaned]
            if target.lower() in self._catalog:
                return self._catalog[target.lower()]

        # 3. Substring match
        for key, item in self._catalog.items():
            if cleaned in key or key in cleaned:
                return item

        # 4. Synonym substring match
        for syn, target in self._synonym_map.items():
            if cleaned in syn or syn in cleaned:
                if target.lower() in self._catalog:
                    return self._catalog[target.lower()]

        return None

    def calculate_nutrition(self, payload: FeedReferenceRequest) -> FeedReferenceResponse:
        """
        Calculates per-kg and total nutritional delivery for quantity_kg:
        total nutrient = per_kg nutrient * quantity_kg
        """
        matched_item = self.find_feed(payload.feed_name)
        if not matched_item:
            available_examples = [
                "Maize Grain", "Maize Silage", "Hybrid Napier", "Sorghum Fodder",
                "Wheat Bran", "Groundnut Cake", "Cottonseed Cake", "Soybean Meal",
                "Paddy Straw", "Wheat Straw", "Concentrate Feed Type II"
            ]
            raise ValueError(
                f"Feed '{payload.feed_name}' not found in authoritative reference database. "
                f"Available examples: {', '.join(available_examples)}"
            )

        dm_pct = matched_item["dry_matter_pct"]
        dm_fraction = dm_pct / 100.0  # e.g. 0.88 for 88% DM
        dm_g_per_kg = round(dm_fraction * 1000.0, 2)  # 880.0 g DM in 1 kg fresh feed

        # Nutrients on 1 kg fresh (as-fed) basis:
        # nutrient_g = (% DM / 100.0) * dry_matter_g
        cp_g = round((matched_item["crude_protein_pct"] / 100.0) * dm_g_per_kg, 2)
        cf_g = round((matched_item["crude_fibre_pct"] / 100.0) * dm_g_per_kg, 2)
        ndf_g = round((matched_item["ndf_pct"] / 100.0) * dm_g_per_kg, 2)
        adf_g = round((matched_item["adf_pct"] / 100.0) * dm_g_per_kg, 2)
        adl_g = round((matched_item["adl_pct"] / 100.0) * dm_g_per_kg, 2) if matched_item["adl_pct"] is not None else None
        starch_g = round((matched_item["starch_pct"] / 100.0) * dm_g_per_kg, 2) if matched_item["starch_pct"] is not None else None
        ee_g = round((matched_item["ether_extract_pct"] / 100.0) * dm_g_per_kg, 2)
        ash_g = round((matched_item["ash_pct"] / 100.0) * dm_g_per_kg, 2)
        
        # Energy and Minerals (ME is MJ/kg DM, Ca/P are g/kg DM)
        energy_mj = round(matched_item["energy_mj_kg"] * dm_fraction, 2)
        ca_g = round(matched_item["calcium_g_kg"] * dm_fraction, 3)
        p_g = round(matched_item["phosphorus_g_kg"] * dm_fraction, 3)

        per_kg_profile = NutrientProfile(
            dry_matter_g=dm_g_per_kg,
            crude_protein_g=cp_g,
            crude_fibre_g=cf_g,
            ndf_g=ndf_g,
            adf_g=adf_g,
            adl_g=adl_g,
            starch_g=starch_g,
            ether_extract_g=ee_g,
            ash_g=ash_g,
            energy_mj=energy_mj,
            calcium_g=ca_g,
            phosphorus_g=p_g
        )

        qty = payload.quantity_kg
        total_profile = NutrientProfile(
            dry_matter_g=round(dm_g_per_kg * qty, 2),
            crude_protein_g=round(cp_g * qty, 2),
            crude_fibre_g=round(cf_g * qty, 2),
            ndf_g=round(ndf_g * qty, 2),
            adf_g=round(adf_g * qty, 2),
            adl_g=round(adl_g * qty, 2) if adl_g is not None else None,
            starch_g=round(starch_g * qty, 2) if starch_g is not None else None,
            ether_extract_g=round(ee_g * qty, 2),
            ash_g=round(ash_g * qty, 2),
            energy_mj=round(energy_mj * qty, 2),
            calcium_g=round(ca_g * qty, 3),
            phosphorus_g=round(p_g * qty, 3)
        )

        percentages_dm = {
            "dry_matter_percent": matched_item["dry_matter_pct"],
            "crude_protein_percent_dm": matched_item["crude_protein_pct"],
            "crude_fibre_percent_dm": matched_item["crude_fibre_pct"],
            "ndf_percent_dm": matched_item["ndf_pct"],
            "adf_percent_dm": matched_item["adf_pct"],
            "adl_percent_dm": matched_item["adl_pct"],
            "starch_percent_dm": matched_item["starch_pct"],
            "ether_extract_percent_dm": matched_item["ether_extract_pct"],
            "ash_percent_dm": matched_item["ash_pct"],
            "metabolizable_energy_mj_kg_dm": matched_item["energy_mj_kg"]
        }

        return FeedReferenceResponse(
            success=True,
            feed_name=payload.feed_name,
            matched_feed_name=matched_item["feed_name"],
            category=matched_item["category"],
            quantity_kg=qty,
            basis="reference",
            per_kg=per_kg_profile,
            total_for_quantity=total_profile,
            nutrient_percentages_dm=percentages_dm,
            source=matched_item["source"],
            disclaimer="Reference nutritional values; actual batch composition may vary."
        )

    def get_all_feeds(self) -> FeedCatalogResponse:
        """Returns the full catalog of feed ingredients for offline frontend caching."""
        items: List[FeedReferenceItem] = []
        for item in self._catalog.values():
            items.append(FeedReferenceItem(**item))

        return FeedCatalogResponse(
            success=True,
            total_feeds=len(items),
            feeds=items
        )


feed_reference_service = FeedReferenceService()
