from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    neighbourhood: str = Field(..., example="Pinheiros")
    room_type: str = Field(..., example="Entire home/apt")
    accommodates: int = Field(..., ge=1, le=20, example=4)
    bathrooms: float = Field(..., ge=0, le=10, example=1.0)
    bedrooms: int = Field(..., ge=0, le=10, example=2)
    beds: int = Field(..., ge=0, le=20, example=2)
    minimum_nights: int = Field(default=1, ge=1, example=2)
    maximum_nights: int = Field(default=365, ge=1, example=365)
    number_of_reviews: int = Field(default=0, ge=0, example=50)
    review_scores_rating: Optional[float] = Field(default=None, ge=0, le=5, example=4.8)
    review_scores_cleanliness: Optional[float] = Field(default=None, ge=0, le=5)
    review_scores_location: Optional[float] = Field(default=None, ge=0, le=5)
    host_is_superhost: bool = Field(default=False, example=True)
    instant_bookable: bool = Field(default=False, example=False)
    calculated_host_listings_count: int = Field(default=1, ge=1, example=1)
    availability_365: int = Field(default=180, ge=0, le=365, example=180)
    amenities: list[str] = Field(
        default=[],
        example=["wifi", "kitchen", "air conditioning"],
    )


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Preço ótimo previsto em R$")
    price_range_low: float = Field(..., description="Limite inferior do intervalo (R$)")
    price_range_high: float = Field(..., description="Limite superior do intervalo (R$)")
    confidence: str = Field(..., description="Confiança da previsão: low/medium/high")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
