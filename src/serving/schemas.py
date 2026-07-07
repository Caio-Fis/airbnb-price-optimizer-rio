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
    local_median_price: Optional[float] = Field(default=None, description="Mediana de preço do bairro+tipo (R$)")
    seasonal_note: Optional[str] = Field(default=None, description="Observação sobre sazonalidade da data alvo")
    seasonal_multiplier: Optional[float] = Field(default=None, description="Fator multiplicativo aplicado pela data (dow × evento, amortecido)")
    revenue_optimal_price: Optional[float] = Field(default=None, description="Preço que maximiza receita esperada (R$)")
    expected_occupancy_pct: Optional[float] = Field(default=None, description="Ocupação esperada no preço ótimo (%)")
    pricing_strategy: Optional[str] = Field(default=None, description="Estratégia: revenue_optimal | premium_positioning | fallback")


class ListingPredictRequest(BaseModel):
    listing_id: int = Field(..., example=821198084644106078)
    target_date: Optional[date] = Field(default=None, example="2025-12-20")


class ListingPredictionResponse(PredictionResponse):
    listing_id: int = Field(..., description="ID do listing no Airbnb")
    listing_name: str = Field(..., description="Nome do listing")
    listing_neighbourhood: str = Field(..., description="Bairro do listing")
    listing_room_type: str = Field(..., description="Tipo de acomodação")
    listing_accommodates: int = Field(..., description="Capacidade de hóspedes")
    listing_current_price: Optional[float] = Field(default=None, description="Preço atual no dataset (R$)")
    latitude: Optional[float] = Field(default=None, description="Latitude do listing")
    longitude: Optional[float] = Field(default=None, description="Longitude do listing")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
