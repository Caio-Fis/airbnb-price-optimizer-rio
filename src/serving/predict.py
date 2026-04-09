"""
Lógica de predição: aplica encoders e retorna preço previsto.

Features incluídas na inferência:
  - Tabular: accommodates, bathrooms, bedrooms, beds, nights, reviews, etc.
  - Amenities: MultiLabelBinarizer (TOP_AMENITIES)
  - Bairro: TargetEncoder → neighbourhood_enc
  - Geo: distâncias Haversine a POIs + metrô (se lat/lon fornecidos)
  - Sazonalidade: hw_seasonal, month, day_of_week, is_weekend (via target_date)
  - Competição local: comp_count, comp_price_p25/p50/p75, comp_price_rank
  - Reviews: velocity, days_since, total, neg_signal (defaults para listing novo)
"""
import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.features.constants import TOP_AMENITIES
from src.features.geo import FIXED_POIS, _haversine, _dist_to_nearest

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
ENCODERS_PATH = Path(os.getenv("ENCODERS_PATH", "models/encoders.joblib"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
FEATURE_NAMES_PATH = PROCESSED_DATA_PATH / "feature_names.joblib"
COMPETITION_STATS_PATH = PROCESSED_DATA_PATH / "competition_stats.joblib"
SEASONAL_DOW_PATH = PROCESSED_DATA_PATH / "hw_seasonal_by_dow.joblib"
METRO_CACHE_PATH = PROCESSED_DATA_PATH / "geo_metro_cache.json"
DEMAND_PARAMS_PATH = PROCESSED_DATA_PATH / "demand_params.joblib"

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
SNAPSHOT_DATE = date(2025, 9, 26)


class Predictor:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_names = None
        self.competition_stats: dict = {}
        self.demand_params: dict = {}
        self.seasonal_by_dow: dict = {}
        self.metro_stations: list = []
        self.model_version = "unknown"
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}")

        if ENCODERS_PATH.exists():
            self.encoders = joblib.load(ENCODERS_PATH)
            logger.info("Encoders loaded")
        else:
            logger.warning(f"Encoders not found at {ENCODERS_PATH}")

        if FEATURE_NAMES_PATH.exists():
            self.feature_names = joblib.load(FEATURE_NAMES_PATH)

        if COMPETITION_STATS_PATH.exists():
            self.competition_stats = joblib.load(COMPETITION_STATS_PATH)
            logger.info(f"Competition stats loaded — {len(self.competition_stats)} groups")

        if DEMAND_PARAMS_PATH.exists():
            self.demand_params = joblib.load(DEMAND_PARAMS_PATH)
            logger.info(f"Demand params loaded — {len(self.demand_params)} segments")

        if SEASONAL_DOW_PATH.exists():
            self.seasonal_by_dow = joblib.load(SEASONAL_DOW_PATH)
            logger.info("Seasonal DOW lookup loaded")

        if METRO_CACHE_PATH.exists():
            with open(METRO_CACHE_PATH) as f:
                self.metro_stations = [tuple(p) for p in json.load(f)]
            logger.info(f"Metro cache loaded — {len(self.metro_stations)} stations")

    def is_ready(self) -> bool:
        return self.model is not None and self.encoders is not None

    def _geo_features(self, lat: Optional[float], lon: Optional[float]) -> dict:
        """Calcula distâncias a POIs. Retorna zeros se lat/lon não fornecidos."""
        row = {}
        if lat is not None and lon is not None:
            lats = np.array([lat])
            lons = np.array([lon])
            for name, (poi_lat, poi_lon) in FIXED_POIS.items():
                row[f"dist_km_{name}"] = float(_haversine(lats, lons, poi_lat, poi_lon)[0])
            if self.metro_stations:
                row["dist_km_metro"] = float(
                    _dist_to_nearest(lats, lons, self.metro_stations)[0]
                )
            else:
                row["dist_km_metro"] = 0.0
        else:
            for name in FIXED_POIS:
                row[f"dist_km_{name}"] = 0.0
            row["dist_km_metro"] = 0.0
        return row

    def _seasonality_features(self, target_date: Optional[date]) -> dict:
        """Retorna features de sazonalidade para a data alvo (ou hoje)."""
        d = target_date or date.today()
        dow = d.weekday()  # 0=Mon, 6=Sun
        return {
            "hw_seasonal": self.seasonal_by_dow.get(dow, 0.0),
            "hw_residual": 0.0,
            "day_of_week": dow,
            "month": d.month,
            "is_weekend": int(dow >= 5),
        }

    def _competition_features(self, neighbourhood: str, room_type: str) -> dict:
        """Lookup de estatísticas locais por bairro+tipo."""
        stats = self.competition_stats.get((neighbourhood, room_type), {})
        return {
            "comp_count": float(stats.get("count", 0)),
            "comp_price_p25": float(stats.get("p25", 0.0)),
            "comp_price_p50": float(stats.get("p50", 0.0)),
            "comp_price_p75": float(stats.get("p75", 0.0)),
            "comp_price_rank": 0.5,  # posição central no mercado por default
        }

    def _build_features(self, req: PredictionRequest) -> np.ndarray:
        row = {
            "accommodates": req.accommodates,
            "bathrooms": req.bathrooms,
            "bedrooms": req.bedrooms,
            "beds": req.beds,
            "minimum_nights": req.minimum_nights,
            "maximum_nights": req.maximum_nights,
            "number_of_reviews": req.number_of_reviews,
            "review_scores_rating": req.review_scores_rating or 4.5,
            "review_scores_cleanliness": req.review_scores_cleanliness or 4.5,
            "review_scores_location": req.review_scores_location or 4.5,
            "calculated_host_listings_count": req.calculated_host_listings_count,
            "availability_365": req.availability_365,
            "host_is_superhost": int(req.host_is_superhost),
            "instant_bookable": int(req.instant_bookable),
            "has_availability": 1,
        }

        # One-hot para room_type (drop_first=True — base: Entire home/apt)
        for rt in ROOM_TYPES[1:]:
            row[f"room_type_{rt}"] = int(req.room_type == rt)

        # Target encoding para bairro
        target_encoder = self.encoders["target_encoder"]
        row["neighbourhood_enc"] = float(
            target_encoder.transform(
                pd.DataFrame({"neighbourhood_cleansed": [req.neighbourhood]})
            ).iloc[0, 0]
        )

        # Amenities
        mlb = self.encoders["mlb"]
        amenity_matrix = mlb.transform([[a.lower() for a in req.amenities]])
        for i, amenity in enumerate(TOP_AMENITIES):
            row[f"amenity_{amenity.replace(' ', '_')}"] = int(amenity_matrix[0, i])

        # Geo features
        row.update(self._geo_features(req.latitude, req.longitude))

        # Sazonalidade
        row.update(self._seasonality_features(req.target_date))

        # Competição local
        row.update(self._competition_features(req.neighbourhood, req.room_type))

        # Reviews — defaults para listing novo (sem histórico)
        row.update({
            "review_velocity": 0.0,
            "days_since_last_review": float((date.today() - SNAPSHOT_DATE).days),
            "total_reviews": float(req.number_of_reviews),
            "n_neg_keywords": 0.0,
            "neg_keyword_ratio": 0.0,
        })

        # Imagens — zeros (listing sem fotos analisadas)
        for score in ["luxury_score", "cleanliness_score", "brightness_score",
                      "professional_photo_score", "modern_style_score"]:
            row[f"clip_{score}"] = 0.0
        for i in range(20):
            row[f"clip_emb_{i}"] = 0.0
        row["object_count"] = 0
        row["yolo_luxury_score"] = 0.0
        row["bed_count"] = req.beds

        df = pd.DataFrame([row])

        # Garantir mesma ordem e conjunto de features do treino
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        return df

    def _revenue_optimal(self, neighbourhood: str, room_type: str) -> tuple[Optional[float], Optional[float], str]:
        """Retorna (revenue_optimal_price, expected_occupancy, strategy) do segmento."""
        from scipy.special import expit
        params = self.demand_params.get((neighbourhood, room_type))
        if not params or params.get("b") is None:
            return None, None, "fallback"

        a = params["a"]
        b = params["b"]
        comp_p50 = params["comp_p50"]
        comp_p75 = params.get("comp_p75", comp_p50 * 1.25)
        strategy = params.get("strategy", "fallback")

        if b < -1:
            grid = np.linspace(0.3, 3.5, 500)
            revenues = grid * comp_p50 * expit(a + b * np.log(np.maximum(grid, 1e-6)))
            best = int(np.argmax(revenues))
            opt_price = float(grid[best] * comp_p50)
            opt_occ = float(expit(a + b * np.log(grid[best])))
        else:
            opt_price = float(comp_p75) if comp_p75 > comp_p50 else comp_p50 * 1.25
            opt_occ = float(expit(a + b * np.log(opt_price / comp_p50))) if comp_p50 > 0 else 0.4

        return round(opt_price, 2), round(opt_occ * 100, 1), strategy

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        from scipy.special import expit as _expit

        X = self._build_features(req)
        log_pred = float(self.model.predict(X)[0])
        market_price = float(np.expm1(log_pred))

        # Confiança baseada no número de reviews
        if req.number_of_reviews >= 20:
            confidence = "high"
        elif req.number_of_reviews >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        # Competição local
        comp = self.competition_stats.get((req.neighbourhood, req.room_type), {})
        local_median = comp.get("p50")

        # Preço ótimo de receita via curva de demanda
        revenue_price, exp_occ, strategy = self._revenue_optimal(req.neighbourhood, req.room_type)

        # Intervalo de confiança: ±15% em torno do market_price
        low = market_price * 0.85
        high = market_price * 1.15

        # Nota de sazonalidade
        seasonal_note = None
        if req.target_date:
            dow_names = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
            dow = req.target_date.weekday()
            seasonal_note = f"{dow_names[dow]}, mês {req.target_date.month}"

        return PredictionResponse(
            predicted_price=round(market_price, 2),
            price_range_low=round(low, 2),
            price_range_high=round(high, 2),
            confidence=confidence,
            local_median_price=round(local_median, 2) if local_median else None,
            seasonal_note=seasonal_note,
            revenue_optimal_price=revenue_price,
            expected_occupancy_pct=exp_occ,
            pricing_strategy=strategy,
        )
