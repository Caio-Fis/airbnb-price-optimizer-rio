"""
Lógica de predição: aplica encoders e retorna preço previsto.
"""
import os
from pathlib import Path

import joblib
import numpy as np
from loguru import logger

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.features.tabular import TOP_AMENITIES

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
ENCODERS_PATH = Path(os.getenv("ENCODERS_PATH", "models/encoders.joblib"))
FEATURE_NAMES_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed")) / "feature_names.joblib"

LOW_CARD_COLS = ["room_type"]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]


class Predictor:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_names = None
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

    def is_ready(self) -> bool:
        return self.model is not None and self.encoders is not None

    def _build_features(self, req: PredictionRequest) -> np.ndarray:
        import pandas as pd

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
        }

        # One-hot encoding para room_type
        for rt in ROOM_TYPES[1:]:  # drop_first=True
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

        # Imagens: sem imagem → zeros (listings novos)
        for score in ["luxury_score", "cleanliness_score", "brightness_score",
                      "professional_photo_score", "modern_style_score"]:
            row[f"clip_{score}"] = 0.0
        for i in range(20):
            row[f"clip_emb_{i}"] = 0.0
        row["object_count"] = 0
        row["yolo_luxury_score"] = 0.0
        row["bed_count"] = req.beds

        df = pd.DataFrame([row])

        # Garantir que todas as features do treino estão presentes
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        return df.values

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        X = self._build_features(req)
        log_pred = float(self.model.predict(X)[0])
        price = float(np.expm1(log_pred))

        # Intervalo de confiança heurístico (±15%)
        low = price * 0.85
        high = price * 1.15

        # Confiança baseada no número de reviews
        if req.number_of_reviews >= 20:
            confidence = "high"
        elif req.number_of_reviews >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictionResponse(
            predicted_price=round(price, 2),
            price_range_low=round(low, 2),
            price_range_high=round(high, 2),
            confidence=confidence,
        )
