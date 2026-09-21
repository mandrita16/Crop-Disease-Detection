from pydantic import BaseModel, Field
from typing import Optional


class TabularInput(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen content")
    P: float = Field(..., ge=0, le=200, description="Phosphorus content")
    K: float = Field(..., ge=0, le=200, description="Potassium content")
    temperature: float = Field(..., ge=-10, le=55)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0, le=500)


class ImagePredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    gradcam_url: str


class TabularPredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    feature_contributions: dict


class FusionResponse(BaseModel):
    image_result: ImagePredictionResponse
    tabular_result: TabularPredictionResponse
    final_prediction: str
    final_confidence: float
    needs_review: bool


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    disease_prediction: Optional[str]
    quality_prediction: Optional[str]
    fusion_prediction: Optional[str]
    needs_review: Optional[bool]

    class Config:
        from_attributes = True
