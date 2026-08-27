"""
Milk Production Yield Estimation Schemas
"""

from typing import Optional
from pydantic import BaseModel, Field


class MilkProductionInput(BaseModel):
    """Input features for predicting daily milk yield (Litres)."""

    # Core Animal & Lactation Features
    Age_Months: float = Field(default=48.0, ge=12.0, le=240.0, description="Age of cow in months")
    Weight_kg: float = Field(default=450.0, ge=100.0, le=1200.0, description="Body weight in kilograms")
    Parity: int = Field(default=2, ge=1, le=15, description="Number of calvings / lactation number")
    Days_in_Milk: float = Field(default=90.0, ge=0.0, le=500.0, description="Days elapsed in current lactation cycle")
    Previous_Week_Avg_Yield: float = Field(default=18.5, ge=0.0, le=80.0, description="Previous week average yield (L)")
    Body_Condition_Score: float = Field(default=3.0, ge=1.0, le=5.0, description="BCS score (1 to 5 scale)")
    Milking_Interval_hrs: float = Field(default=12.0, ge=6.0, le=24.0, description="Hours between milking sessions")

    # Feed & Water Management
    Feed_Quantity_kg: float = Field(default=22.0, ge=1.0, le=60.0, description="Total feed intake per day in kg")
    Feeding_Frequency: float = Field(default=2.0, ge=1.0, le=6.0, description="Feeding events per day")
    Water_Intake_L: float = Field(default=75.0, ge=10.0, le=200.0, description="Daily water consumption in litres")
    Grazing_Duration_hrs: float = Field(default=4.0, ge=0.0, le=24.0, description="Pasture grazing hours per day")

    # Activity & Behavior
    Walking_Distance_km: float = Field(default=1.5, ge=0.0, le=30.0, description="Daily walking distance in km")
    Rumination_Time_hrs: float = Field(default=8.0, ge=0.0, le=20.0, description="Rumination duration in hours")
    Resting_Hours: float = Field(default=10.0, ge=0.0, le=24.0, description="Resting hours per day")

    # Environmental & Housing Conditions
    Ambient_Temperature_C: float = Field(default=24.0, ge=-20.0, le=55.0, description="Ambient temperature in °C")
    Humidity_percent: float = Field(default=60.0, ge=5.0, le=100.0, description="Relative humidity percentage")
    Housing_Score: float = Field(default=4.0, ge=1.0, le=5.0, description="Housing quality index (1-5 scale)")

    # Vaccination Status (0 = No, 1 = Yes)
    FMD_Vaccine: int = Field(default=1, ge=0, le=1, description="Foot-and-Mouth Disease vaccinated")
    Brucellosis_Vaccine: int = Field(default=1, ge=0, le=1, description="Brucellosis vaccinated")
    HS_Vaccine: int = Field(default=1, ge=0, le=1, description="Haemorrhagic Septicaemia vaccinated")
    BQ_Vaccine: int = Field(default=1, ge=0, le=1, description="Black Quarter vaccinated")
    Anthrax_Vaccine: int = Field(default=0, ge=0, le=1, description="Anthrax vaccinated")
    IBR_Vaccine: int = Field(default=0, ge=0, le=1, description="Infectious Bovine Rhinotracheitis vaccinated")
    BVD_Vaccine: int = Field(default=0, ge=0, le=1, description="Bovine Viral Diarrhea vaccinated")
    Rabies_Vaccine: int = Field(default=0, ge=0, le=1, description="Rabies vaccinated")

    # Categorical Descriptors
    Cattle_ID: str = Field(default="COW_001", description="Animal identifier")
    Breed: str = Field(default="Holstein_Friesian", description="Breed or crossbreed")
    Region: str = Field(default="Northern", description="Geographic farm region")
    Country: str = Field(default="India", description="Country")
    Climate_Zone: str = Field(default="Tropical", description="Climate classification")
    Management_System: str = Field(default="Intensive", description="Management system (Intensive/Semi-Intensive/Extensive)")
    Lactation_Stage: str = Field(default="Mid", description="Lactation stage (Early/Mid/Late)")
    Feed_Type: str = Field(default="TMR", description="Diet type (TMR/Silage/Green Fodder/Concentrate)")
    Season: str = Field(default="Monsoon", description="Current season")
    Date: str = Field(default="2026-08-26", description="Measurement date")
    Farm_ID: str = Field(default="FARM_01", description="Farm unit identifier")


class MilkProductionPredictionResponse(BaseModel):
    """Output schema for milk yield regression."""
    predicted_milk_yield_litres: float = Field(..., description="Estimated daily milk production in litres")
    target_unit: str = Field(default="Litres / Day", description="Unit of measurement")
    model_r2_score: float = Field(default=0.946193, description="Model validation R2 determination coefficient")
    features_received: int = Field(default=36, description="Number of input parameters processed")
