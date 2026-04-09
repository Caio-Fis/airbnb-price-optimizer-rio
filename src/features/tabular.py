"""
Feature engineering tabular:
- Limpeza e encoding de categóricas
- Target Encoding para bairros
- MultiLabelBinarizer para amenities
- Holt-Winters para sazonalidade via calendário
"""
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.features.constants import TOP_AMENITIES

RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
CITY = os.getenv("INSIDE_AIRBNB_CITY", "rio-de-janeiro")

LOW_CARD_COLS = ["room_type", "cancellation_policy", "host_is_superhost"]
HIGH_CARD_COLS = ["neighbourhood_cleansed"]  # target encoding
BINARY_COLS = ["instant_bookable", "has_availability"]
NUMERIC_COLS = [
    "accommodates", "bathrooms", "bedrooms", "beds",
    "minimum_nights", "maximum_nights",
    "number_of_reviews", "review_scores_rating",
    "review_scores_cleanliness", "review_scores_location",
    "calculated_host_listings_count", "availability_365",
]


class TabularFeaturePipeline:
    def __init__(self):
        self.target_encoder = TargetEncoder(smoothing=10, min_samples_leaf=5)
        self.mlb = MultiLabelBinarizer(classes=TOP_AMENITIES)
        self._fitted = False

    def _load_raw(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED_DATA_PATH / "listings_set2025.parquet")

    def _clean_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
        df = df[(df["price"] >= 10) & (df["price"] <= 50_000)].copy()
        df["log_price"] = np.log1p(df["price"])
        return df

    def _parse_amenities(self, df: pd.DataFrame) -> pd.DataFrame:
        def parse(raw):
            if pd.isna(raw):
                return []
            items = re.findall(r'"([^"]+)"', str(raw))
            return [i.lower() for i in items]

        df["amenities_list"] = df["amenities"].apply(parse)
        return df

    def _encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in BINARY_COLS:
            if col in df.columns:
                df[col] = (df[col].str.strip().str.lower() == "t").astype(int)
        return df

    def _encode_low_card(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in LOW_CARD_COLS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        return df

    def _fill_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
        return df

    def run(self, fit: bool = True) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        df = self._load_raw()
        df = self._clean_price(df)
        df = self._parse_amenities(df)
        df = self._encode_binary(df)
        df = self._encode_low_card(df)
        df = self._fill_numerics(df)

        # MultiLabelBinarizer para amenities
        amenity_matrix = self.mlb.fit_transform(df["amenities_list"]) if fit \
            else self.mlb.transform(df["amenities_list"])
        amenity_df = pd.DataFrame(
            amenity_matrix,
            columns=[f"amenity_{a.replace(' ', '_')}" for a in self.mlb.classes_],
            index=df.index,
        )
        df = pd.concat([df, amenity_df], axis=1)

        # Target Encoding para bairro
        if fit:
            df["neighbourhood_enc"] = self.target_encoder.fit_transform(
                df[["neighbourhood_cleansed"]], df["log_price"]
            )
        else:
            df["neighbourhood_enc"] = self.target_encoder.transform(
                df[["neighbourhood_cleansed"]]
            )

        # Salvar encoders
        if fit:
            encoder_path = PROCESSED_DATA_PATH / "encoders.joblib"
            joblib.dump(
                {"target_encoder": self.target_encoder, "mlb": self.mlb},
                encoder_path,
            )
            logger.info(f"Encoders saved to {encoder_path}")

        output_path = PROCESSED_DATA_PATH / "tabular_features.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Tabular features saved to {output_path} — shape: {df.shape}")
        return str(output_path)


class SeasonalityPipeline:
    """
    Usa o calendário do Inside Airbnb para extrair features de sazonalidade
    via Holt-Winters (Triple Exponential Smoothing).
    """

    def run(self) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

        # Preço do calendário é 100% nulo nesse scrape — usar disponibilidade diária
        def _load_avail(snapshot: str) -> pd.Series:
            df = pd.read_parquet(
                PROCESSED_DATA_PATH / f"calendar_{snapshot}.parquet",
                columns=["date", "available"],
            )
            df["is_avail"] = (df["available"] == "t").astype("int8")
            return df.groupby("date")["is_avail"].mean().mul(100).sort_index()

        agg_jun = _load_avail("jun2025")
        agg_set = _load_avail("set2025")
        daily_avail = agg_jun.combine_first(agg_set)
        daily_avail.update(agg_set)
        daily_avail = daily_avail.asfreq("D").interpolate()

        seasonality_features = self._extract_hw_features(daily_avail)

        output_path = PROCESSED_DATA_PATH / "seasonality_features.parquet"
        seasonality_features.to_parquet(output_path)
        logger.info(f"Seasonality features saved to {output_path}")
        return str(output_path)

    def _extract_hw_features(self, series: pd.Series) -> pd.DataFrame:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=7,  # sazonalidade semanal
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)

        features = pd.DataFrame(index=series.index)
        features["hw_level"] = fit.level
        features["hw_trend"] = fit.trend
        features["hw_seasonal"] = fit.season
        features["hw_fitted"] = fit.fittedvalues
        features["hw_residual"] = series - fit.fittedvalues

        # Sazonalidade mensal via dummies
        features["month"] = features.index.month
        features["day_of_week"] = features.index.dayofweek
        features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)

        logger.info(
            f"Holt-Winters fitted. AIC: {fit.aic:.2f} | "
            f"Seasonal amplitude: {features['hw_seasonal'].std():.2f}"
        )

        # Salvar lookup sazonal por dia da semana para inferência prospectiva
        seasonal_by_dow = (
            features.groupby("day_of_week")["hw_seasonal"].mean().to_dict()
        )
        joblib.dump(seasonal_by_dow, PROCESSED_DATA_PATH / "hw_seasonal_by_dow.joblib")
        logger.info("Seasonal lookup by day_of_week saved.")

        return features
