from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    neighbourhood: str = Field(..., example="Copacabana")
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
    # Localização — permite calcular distâncias geo reais
    latitude: Optional[float] = Field(default=None, example=-22.9711)
    longitude: Optional[float] = Field(default=None, example=-43.1822)
    # Data alvo — ajusta sazonalidade semanal/mensal na previsão
    target_date: Optional[date] = Field(default=None, example="2025-12-20")


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Benchmark de mercado: o que listings similares cobram (R$)")
    price_range_low: float = Field(..., description="Limite inferior do intervalo de mercado (R$)")
    price_range_high: float = Field(..., description="Limite superior do intervalo de mercado (R$)")
    confidence: str = Field(..., description="Confiança da previsão: low/medium/high")
    local_median_price: Optional[float] = Field(default=None, description="Mediana de preço do bairro+tipo (R$)")
    seasonal_note: Optional[str] = Field(default=None, description="Observação sobre sazonalidade da data alvo")
    revenue_optimal_price: Optional[float] = Field(default=None, description="Preço que maximiza receita esperada (R$)")
    expected_occupancy_pct: Optional[float] = Field(default=None, description="Ocupação esperada no preço ótimo (%)")
    pricing_strategy: Optional[str] = Field(default=None, description="Estratégia: revenue_optimal | premium_positioning | fallback")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
