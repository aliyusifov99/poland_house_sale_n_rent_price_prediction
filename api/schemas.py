from pydantic import BaseModel
from typing import Optional

class PropertyData(BaseModel):
    # We list ALL features used in training.
    # The names must match the columns in your CSV exactly.
    city: str
    type: str
    squareMeters: float
    rooms: float
    floor: Optional[float] = 1.0
    floorCount: Optional[float] = 1.0
    buildYear: Optional[float] = 1980.0
    centreDistance: float
    poiCount: float
    schoolDistance: float
    clinicDistance: float
    postOfficeDistance: float
    kindergartenDistance: float
    restaurantDistance: float
    collegeDistance: float
    pharmacyDistance: float
    ownership: str
    buildingMaterial: Optional[str] = "brick"
    condition: Optional[str] = "unknown"
    hasParkingSpace: int = 0
    hasBalcony: int = 0
    hasElevator: int = 0
    hasSecurity: int = 0
    hasStorageRoom: int = 0
    
    # We don't need boolean features (parking, etc) if you didn't include them 
    # in the CATEGORICAL_FEATURES list in training. 
    # If you did, add them here.