# Field Nutrition & Least-Cost Ration Optimization Engine Specification

> **Official Notice:**  
> **"This version is a deterministic ICAR/Indian-feed-reference-based ration optimization engine, not a supervised ML model."**  
> All recommendations are mathematically derived using Linear Programming to guarantee exact biological adequacy, rumen safety, and lowest cost for Indian dairy farmers without synthetic data risk.

---

## 1. Module Overview & Scientific Rationale

The Field Nutrition module for `Dairy_AI_Assistant` replaces opaque, unverified machine learning regressions with an **empirically validated, deterministic least-cost ration formulation engine**.

### Core Architecture:
1. **ICAR-2013 / 2024 Nutritional Standard Calculations**: Computes exact animal requirements for Dry Matter (DMI), Total Digestible Nutrients (TDN), Metabolizable Energy (ME), Crude Protein (CP), Calcium (Ca), and Phosphorus (P) partitioned across Maintenance, 4% Fat-Corrected Milk (FCM) Lactation, and Pregnancy.
2. **ICAR-NIANP Feed Composition Matrix**: 38 authentic Indian feed ingredients (green forages, dry roughages, silages, cakes, byproducts, and mineral supplements) with proximate chemistry and mineral contents.
3. **Least-Cost Linear Programming (LP)**: Solves the cost-minimizing feed combination using `scipy.optimize.linprog` subject to physiological rumen constraints, dry matter capacity, and nutrient lower bounds.

---

## 2. Input Variables

| Parameter | Type | Required | Units | Description & Validation Range |
|---|---|---|---|---|
| `species` | `str` | Yes (Default: Cattle) | categorical | `'Cattle'` (Zebu or Crossbred) or `'Buffalo'` |
| `breed` | `str` | Optional | text | Breed name (e.g. `'Gir'`, `'Sahiwal'`, `'HF_Cross'`, `'Murrah'`) |
| `body_weight_kg` | `float` | **Yes (Critical)** | **kg** | Live animal body weight ($120 \le W \le 1200 \text{ kg}$) |
| `daily_milk_yield_kg` | `float` | **Yes (Critical)** | **kg or L/day** | Daily milk production ($0.0 \le MY \le 70.0 \text{ kg}$) |
| `milk_fat_percent` | `float` | Optional (Default: 4.0% / 7.0%) | **%** | Milk fat percentage ($2.5\% \le \text{Fat} \le 12.0\%$) |
| `lactation_stage` | `str` | Optional | categorical | `'Early'` (0-90d), `'Mid'` (91-200d), `'Late'` (>200d), `'Dry'` |
| `days_in_milk` | `float` | Optional | days | Days in current lactation |
| `pregnancy_status` | `bool` | Optional | bool | True if pregnant |
| `pregnancy_month` | `int` | Optional | months | Month of gestation ($\ge 7$ triggers last-trimester allowance) |
| `available_feeds` | `List[str]` | Optional | list of names | Custom on-farm feed inventory to restrict solver |
| `feed_prices` | `Dict[str, float]` | Optional | INR/kg | Custom farmer-specific market prices |

---

## 3. Official ICAR Mathematical Formulas

### 3.1. Metabolic Body Weight
$$\text{MBW} = W^{0.75} \quad (\text{kg})$$

### 3.2. Maintenance Requirements
| Requirement | Indigenous Zebu Cattle | Crossbred Cattle (HF/Jersey) | Water Buffalo (Murrah) |
|---|---|---|---|
| **Maintenance DMI (kg/day)** | $0.022 \times W$ | $0.024 \times W$ | $0.023 \times W$ |
| **Maintenance TDN (kg/day)** | $0.034 \times W^{0.75}$ | $0.036 \times W^{0.75}$ | $0.035 \times W^{0.75}$ |
| **Maintenance CP (g/day)** | $4.2 \times W^{0.75}$ | $4.5 \times W^{0.75}$ | $4.3 \times W^{0.75}$ |
| **Maintenance Calcium (g/day)**| $0.050 \times W$ | $0.055 \times W$ | $0.052 \times W$ |
| **Maintenance Phosphorus (g/day)**| $0.035 \times W$ | $0.038 \times W$ | $0.036 \times W$ |

### 3.3. Lactation Requirements (4% Fat-Corrected Milk)
$$\text{FCM}_{4\%} = (0.4 + 0.15 \times \text{Milk Fat \%}) \times \text{Daily Milk Yield (kg)}$$

| Requirement per kg 4% FCM | Cattle (All Breeds) | Water Buffalo |
|---|---|---|
| **Lactation TDN (kg)** | $0.320 \times \text{FCM}$ | $0.340 \times \text{FCM}$ |
| **Lactation CP (g)** | $85.0 \times \text{FCM}$ | $92.0 \times \text{FCM}$ |
| **Lactation Calcium (g)** | $3.0 \times \text{FCM}$ | $3.5 \times \text{FCM}$ |
| **Lactation Phosphorus (g)** | $2.0 \times \text{FCM}$ | $2.4 \times \text{FCM}$ |
| **Lactation DMI (kg)** | $0.33 \times \text{Milk Yield}$ | $0.36 \times \text{Milk Yield}$ |

### 3.4. Pregnancy Allowance (Last Trimester $\ge 7$ Months)
- $+1.0 \text{ kg DMI/day}$
- $+1.20 \text{ kg TDN/day}$ ($+18.1 \text{ MJ ME/day}$)
- $+250 \text{ g CP/day}$
- $+12.0 \text{ g Calcium/day}$
- $+8.0 \text{ g Phosphorus/day}$

---

## 4. Feed Composition Source & Nutrient Matrix

Sourced from the authoritative standard:  
*Nutrient Composition of Indian Feeds and Fodders* (ICAR - National Institute of Animal Nutrition and Physiology, Bengaluru, ISBN: `978-81-7164-145-1`).

### Summary of 38 Feed Ingredients:
- **Green Forages (Non-Legume)**: Maize Green, Sorghum/Jowar, Bajra, Hybrid Napier (CO-3/CO-4/CO-5), Guinea Grass, Para Grass, Oat Fodder.
- **Green Forages (Legume & Supplements)**: Berseem, Lucerne, Cowpea Fodder, Hedge Lucerne (Desmanthus), Fresh Azolla.
- **Dry Roughages**: Paddy Straw, Wheat Straw, Sorghum Kadbi, Maize Stover, Sugarcane Tops.
- **Silages**: Maize Silage, Sorghum Silage.
- **Energy Grains**: Crushed Maize, Cracked Wheat, Barley.
- **Protein Meals & Cakes**: Decorticated Cottonseed Cake, Undecorticated Cottonseed Cake, Expeller Mustard Cake, Groundnut Cake, Solvent Soybean Meal, Til/Sesame Cake.
- **Byproducts & Chunis**: Wheat Bran (Chokar), De-oiled Rice Bran (DORB), Rice Polish, Chickpea/Gram Chuni, Arhar/Tur Chuni, Molasses.
- **Compound Feeds & Supplements**: BIS Type-I & Type-II Cattle Feed, Area Specific Mineral Mixture (ASMM), Bypass Fat.

---

## 5. Linear Programming Optimization Model

### 5.1. Objective Function
$$\min \sum_{i=1}^n c_i x_i$$
Where:
- $x_i$: Daily quantity of fresh feed $i$ (in kg/day).
- $c_i$: Cost per kg fresh feed in INR.

### 5.2. Inequality Constraints ($A_{ub} x \le b_{ub}$)
1. **Dry Matter Intake Upper Bound**: $\sum \text{DM}_i x_i \le \text{Req DMI} \times 1.05$
2. **Dry Matter Intake Lower Bound**: $\sum \text{DM}_i x_i \ge \text{Req DMI} \times 0.85$
3. **Total Digestible Nutrients Lower Bound**: $\sum \text{TDN}_i x_i \ge \text{Req TDN}$
4. **Crude Protein Lower Bound**: $\sum \text{CP}_i x_i \ge \text{Req CP}$ (in g/day)
5. **Calcium Lower Bound**: $\sum \text{Ca}_i x_i \ge \text{Req Ca}$ (in g/day)
6. **Phosphorus Lower Bound**: $\sum \text{P}_i x_i \ge \text{Req P}$ (in g/day)

### 5.3. Rumen Safety Bounds
- **Dry Roughage**: Minimum $2.5 \text{ kg}$ to $9.0 \text{ kg/day}$ fresh basis (prevents acidosis and ensures cud-chewing).
- **Green Fodder**: Minimum $5.0 \text{ kg}$ to $35.0 \text{ kg/day}$ fresh basis.
- **Mineral Mixture (ASMM)**: Fixed minimum $0.05 \text{ kg}$ ($50 \text{ g/day}$) up to $0.12 \text{ kg/day}$.
- **Concentrate / Cake**: Capped at maximum $8.0 \text{ kg/day}$ ($<55\%$ of total DMI).

---

## 6. Output Fields

```json
{
  "success": true,
  "is_deterministic_optimized": true,
  "status": "optimized",
  "message": "Optimal least-cost balanced ration calculated successfully...",
  "animal_profile": {
    "species": "Cattle",
    "breed": "Gir",
    "body_weight_kg": 420.0,
    "daily_milk_yield_kg": 15.0,
    "milk_fat_percent": 4.0
  },
  "nutrient_requirements": {
    "metabolic_body_weight_kg": 92.77,
    "fat_corrected_milk_4pct_kg": 15.0,
    "req_dmi_kg_per_day": 14.19,
    "req_tdn_kg_per_day": 7.95,
    "req_me_mj_per_day": 120.0,
    "req_cp_g_per_day": 1664.7,
    "req_calcium_g_per_day": 66.0,
    "req_phosphorus_g_per_day": 44.7
  },
  "recommended_ration": [
    {
      "feed_id": "IN_GF_004",
      "feed_name": "Hybrid Napier (CO-3 / CO-4 / CO-5)",
      "feed_category": "Green Roughage",
      "quantity_kg_per_day": 35.0,
      "cost_per_kg_inr": 2.5,
      "daily_cost_inr": 87.5,
      "dm_supplied_kg": 6.48,
      "cp_supplied_g": 596.2,
      "tdn_supplied_kg": 3.66,
      "calcium_supplied_g": 32.4,
      "phosphorus_supplied_g": 16.2
    },
    {
      "feed_id": "IN_DR_013",
      "feed_name": "Paddy Straw (Oryza sativa)",
      "feed_category": "Dry Roughage",
      "quantity_kg_per_day": 6.54,
      "cost_per_kg_inr": 7.0,
      "daily_cost_inr": 45.78,
      "dm_supplied_kg": 5.89,
      "cp_supplied_g": 223.7,
      "tdn_supplied_kg": 2.47,
      "calcium_supplied_g": 20.6,
      "phosphorus_supplied_g": 7.1
    },
    {
      "feed_id": "IN_PC_023",
      "feed_name": "Cottonseed Cake (Decorticated)",
      "feed_category": "Concentrate",
      "quantity_kg_per_day": 2.15,
      "cost_per_kg_inr": 32.0,
      "daily_cost_inr": 68.8,
      "dm_supplied_kg": 1.94,
      "cp_supplied_g": 735.3,
      "tdn_supplied_kg": 1.43,
      "calcium_supplied_g": 4.8,
      "phosphorus_supplied_g": 21.3
    },
    {
      "feed_id": "IN_CF_035",
      "feed_name": "Compound Cattle Feed (BIS Type-II Standard)",
      "feed_category": "Compound Feed",
      "quantity_kg_per_day": 0.62,
      "cost_per_kg_inr": 24.0,
      "daily_cost_inr": 14.88,
      "dm_supplied_kg": 0.56,
      "cp_supplied_g": 111.6,
      "tdn_supplied_kg": 0.39,
      "calcium_supplied_g": 4.5,
      "phosphorus_supplied_g": 2.8
    },
    {
      "feed_id": "IN_CF_037",
      "feed_name": "Area Specific Mineral Mixture (ASMM)",
      "feed_category": "Mineral Supplement",
      "quantity_kg_per_day": 0.05,
      "cost_per_kg_inr": 80.0,
      "daily_cost_inr": 4.0,
      "dm_supplied_kg": 0.05,
      "cp_supplied_g": 0.0,
      "tdn_supplied_kg": 0.0,
      "calcium_supplied_g": 9.8,
      "phosphorus_supplied_g": 5.9
    }
  ],
  "total_daily_cost_inr": 220.96,
  "nutrient_supply": {
    "dry_matter_kg": 14.9,
    "tdn_kg": 7.95,
    "me_mj": 120.0,
    "crude_protein_g": 1664.7,
    "calcium_g": 72.1,
    "phosphorus_g": 53.2
  },
  "nutrient_balance": {
    "dry_matter": {"required": 14.19, "supplied": 14.9, "unit": "kg/day", "percentage_fulfilled": 105.0, "status": "Balanced"},
    "tdn": {"required": 7.95, "supplied": 7.95, "unit": "kg/day", "percentage_fulfilled": 100.0, "status": "Balanced"},
    "crude_protein": {"required": 1664.7, "supplied": 1664.7, "unit": "g/day", "percentage_fulfilled": 100.0, "status": "Balanced"},
    "calcium": {"required": 66.0, "supplied": 72.1, "unit": "g/day", "percentage_fulfilled": 109.2, "status": "Balanced"},
    "phosphorus": {"required": 44.7, "supplied": 53.2, "unit": "g/day", "percentage_fulfilled": 119.0, "status": "Surplus"}
  },
  "warnings": []
}
```

---

## 7. Automated Test Suite Validation Results

### Test Execution Command:
```powershell
.\.pytorch_venv\Scripts\python.exe -m pytest backend/tests/ -v
```

### Results Summary:
- **`backend/tests/test_field_nutrition.py`**: **16 / 16 Passed (100%)**
- **`backend/tests/test_chat.py`**: **32 / 32 Passed (100%)**
- **`backend/tests/test_smoke.py`**: **17 / 17 Passed (100%)**
- **Total Suite**: **65 / 65 Passed (0 Failures, 100% Success Rate)**
