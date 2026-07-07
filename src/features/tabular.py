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
    "calculated_host_listings_count", "availability_365",
]


class TabularFeaturePipeline:
    def __init__(self):
        self.target_encoder = TargetEncoder(smoothing=10, min_samples_leaf=5)
        self.mlb = MultiLabelBinarizer(classes=TOP_AMENITIES)
        self._fitted = False
        self._price_p99: float = 0.0

    def _load_raw(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED_DATA_PATH / "listings_set2025.parquet")

    def _clean_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
        df = df[(df["price"] >= 10) & (df["price"] <= 50_000)].copy()
        # Winsorização P1–P99 para reduzir influência de outliers no RMSE
        p1 = df["price"].quantile(0.01)
        p99 = df["price"].quantile(0.99)
        df = df[(df["price"] >= p1) & (df["price"] <= p99)].copy()
        self._price_p99 = float(p99)
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
                {"target_encoder": self.target_encoder, "mlb": self.mlb,
                 "price_p99": self._price_p99},
                encoder_path,
            )
            logger.info(f"Encoders saved to {encoder_path} (price_p99={self._price_p99:.0f})")

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


class SeasonalFactorsPipeline:
    """
    Fatores multiplicativos de preço por data, derivados da ocupação real
    (1 − disponibilidade) dos calendários Jun+Set/2025 — cobertura até Set/2026,
    incluindo Réveillon 2025/26 e Carnaval/2026.

    Premissas documentadas:
      - ocupação é proxy de demanda; índice normalizado para média 1.0
      - multiplicador = (dow × event) ** DAMPENING (elasticidade parcial
        preço↔ocupação), limitado a [CLIP_LOW, CLIP_HIGH]
      - fator MENSAL não é estimado: datas longe do scrape têm disponibilidade
        alta só porque as reservas ainda não aconteceram (artefato de horizonte),
        e com 15 meses de calendário o efeito de mês não é separável do horizonte
      - fatores de EVENTO são medidos contra a linha de base local (±21 dias,
        excluindo o próprio evento), o que cancela o artefato de horizonte
    """

    DAMPENING = 0.5
    CLIP_LOW, CLIP_HIGH = 0.8, 2.5

    def run(self) -> str:
        def _load_avail(snapshot: str) -> pd.Series:
            df = pd.read_parquet(
                PROCESSED_DATA_PATH / f"calendar_{snapshot}.parquet",
                columns=["date", "available"],
            )
            df["is_avail"] = (df["available"] == "t").astype("int8")
            return df.groupby("date")["is_avail"].mean().sort_index()

        agg_jun = _load_avail("jun2025")
        agg_set = _load_avail("set2025")
        daily_avail = agg_jun.combine_first(agg_set)
        daily_avail.update(agg_set)
        daily_avail.index = pd.to_datetime(daily_avail.index)
        daily_avail = daily_avail.asfreq("D").interpolate()

        occ = 1.0 - daily_avail  # proxy de ocupação diária (0-1)
        occ_idx = occ / occ.mean()  # normalizado: média 1.0

        # Dia da semana: distribuído uniformemente ao longo do horizonte → sem viés
        dow_factor = occ_idx.groupby(occ_idx.index.dayofweek).mean().round(4).to_dict()

        import holidays as holidays_lib
        years = sorted(occ_idx.index.year.unique())
        br_rj = holidays_lib.Brazil(subdiv="RJ", years=years, categories=("public", "optional"))

        idx = occ_idx.index
        reveillon_mask = ((idx.month == 12) & (idx.day >= 27)) | ((idx.month == 1) & (idx.day <= 2))

        carnival_days = {d for d, name in br_rj.items() if "Carnival" in name}
        carnival_windows = set()
        for d in carnival_days:
            for offset in range(-2, 3):
                carnival_windows.add(pd.Timestamp(d) + pd.Timedelta(days=offset))
        carnaval_mask = idx.isin(list(carnival_windows))

        holiday_dates = {pd.Timestamp(d) for d in br_rj if pd.Timestamp(d) not in carnival_windows}
        feriado_mask = idx.isin(list(holiday_dates)) & ~reveillon_mask

        def _event_index(mask) -> float | None:
            """Ocupação no evento vs linha de base local (±21 dias, sem o evento).

            A razão local cancela o artefato de horizonte de reserva.
            """
            if not mask.any():
                return None
            ratios = []
            for day in idx[mask]:
                lo, hi = day - pd.Timedelta(days=21), day + pd.Timedelta(days=21)
                window = occ_idx[(idx >= lo) & (idx <= hi) & ~mask]
                if len(window) >= 14:
                    ratios.append(occ_idx[day] / window.mean())
            return round(float(np.mean(ratios)), 4) if ratios else None

        factors = {
            "dow": dow_factor,
            "events": {
                "reveillon": _event_index(reveillon_mask),
                "carnaval": _event_index(carnaval_mask),
                "feriado": _event_index(feriado_mask),
            },
            "dampening": self.DAMPENING,
            "clip": [self.CLIP_LOW, self.CLIP_HIGH],
        }

        output_path = PROCESSED_DATA_PATH / "seasonal_factors.joblib"
        joblib.dump(factors, output_path)
        logger.info(f"Seasonal factors salvos em {output_path}: {factors}")
        return str(output_path)
