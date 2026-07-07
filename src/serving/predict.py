"""
Lógica de predição: aplica encoders e retorna preço previsto.

Features incluídas na inferência:
  - Tabular: accommodates, bathrooms, bedrooms, beds, nights, etc.
  - Amenities: MultiLabelBinarizer (TOP_AMENITIES)
  - Bairro: TargetEncoder → neighbourhood_enc
  - Geo: distâncias Haversine a POIs + metrô (se lat/lon fornecidos)
  - Sazonalidade: hw_seasonal, month, day_of_week, is_weekend (via target_date)
  - Competição local: comp_count, comp_price_p25/p50/p75, comp_price_rank
"""
import json
import os
from datetime import date
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.serving.schemas import PredictionRequest, PredictionResponse, ListingPredictionResponse
from src.features.constants import TOP_AMENITIES
from src.features.demand import _optimal_price
from src.features.geo import (
    FIXED_POIS, ORLA_POINTS, POI_CATEGORIES,
    _haversine, _dist_to_nearest, poi_cache_path, poi_stats,
)
from src.features.bairro import IPS_CSV, IPS_FEATURES, load_ips_lookup, normalize_bairro

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
ENCODERS_PATH = Path(os.getenv("ENCODERS_PATH", "models/encoders.joblib"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
FEATURE_NAMES_PATH = PROCESSED_DATA_PATH / "feature_names.joblib"
COMPETITION_STATS_PATH = PROCESSED_DATA_PATH / "competition_stats.joblib"
SEASONAL_DOW_PATH = PROCESSED_DATA_PATH / "hw_seasonal_by_dow.joblib"
METRO_CACHE_PATH = PROCESSED_DATA_PATH / "geo_metro_cache.json"
DEMAND_PARAMS_PATH = PROCESSED_DATA_PATH / "demand_params.joblib"
TRAINING_STATS_PATH = PROCESSED_DATA_PATH / "training_stats.joblib"
PREDICTION_INTERVALS_PATH = PROCESSED_DATA_PATH / "prediction_intervals.joblib"
SEASONAL_FACTORS_PATH = PROCESSED_DATA_PATH / "seasonal_factors.joblib"
LISTINGS_SLIM_PATH = PROCESSED_DATA_PATH / "listings_slim.parquet"

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]


def _rank_from_percentiles(price: float, p25: float, p50: float, p75: float) -> float:
    """Estima o percentile rank de um preço dado P25/P50/P75 do segmento."""
    if p25 <= 0 or p50 <= 0 or p75 <= p25:
        return 0.5
    if price <= p25:
        return max(0.0, 0.25 * price / p25)
    elif price <= p50:
        return 0.25 + 0.25 * (price - p25) / (p50 - p25)
    elif price <= p75:
        return 0.50 + 0.25 * (price - p50) / (p75 - p50)
    else:
        return min(0.75 + 0.25 * (price - p75) / max(p75 * 0.5, 1.0), 1.0)


class Predictor:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_names = None
        self.competition_stats: dict = {}
        self.demand_params: dict = {}
        self.seasonal_by_dow: dict = {}
        self.metro_stations: list = []
        self.poi_coords: dict[str, list] = {}
        self.ips_lookup: dict[str, dict] = {}
        self.ips_medians: dict[str, float] = {}
        self.seasonal_factors: dict = {}
        self.clip_lookup = None
        self.model_version = "unknown"
        self.training_stats: dict = {}
        self.prediction_intervals: dict = {}
        self.price_p99: Optional[float] = None
        self.listings_lookup = None
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}")

        if ENCODERS_PATH.exists():
            self.encoders = joblib.load(ENCODERS_PATH)
            self.price_p99 = self.encoders.get("price_p99")
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

        for category in POI_CATEGORIES:
            cache = poi_cache_path(category)
            if cache.exists():
                with open(cache) as f:
                    self.poi_coords[category] = [tuple(p) for p in json.load(f)]
        if self.poi_coords:
            logger.info(
                f"POI caches loaded — {len(self.poi_coords)} categorias: "
                f"{ {c: len(p) for c, p in self.poi_coords.items()} }"
            )

        clip_path = PROCESSED_DATA_PATH / "clip_features.parquet"
        if clip_path.exists():
            self.clip_lookup = pd.read_parquet(clip_path).set_index("listing_id")
            logger.info(f"CLIP features loaded — {len(self.clip_lookup)} listings com foto")

        if SEASONAL_FACTORS_PATH.exists():
            self.seasonal_factors = joblib.load(SEASONAL_FACTORS_PATH)
            logger.info(f"Seasonal factors loaded: {self.seasonal_factors.get('events')}")

        if IPS_CSV.exists():
            self.ips_lookup = load_ips_lookup()
            self.ips_medians = {
                feat: float(np.median([v[feat] for v in self.ips_lookup.values()]))
                for feat in IPS_FEATURES
            }
            logger.info(f"IPS lookup loaded — {len(self.ips_lookup)} bairros")

        if TRAINING_STATS_PATH.exists():
            self.training_stats = joblib.load(TRAINING_STATS_PATH)
            logger.info(f"Training stats loaded: {self.training_stats}")

        if PREDICTION_INTERVALS_PATH.exists():
            self.prediction_intervals = joblib.load(PREDICTION_INTERVALS_PATH)
            logger.info(
                f"Prediction intervals loaded: "
                f"P10={self.prediction_intervals.get('p10_pct', -0.15):.2%} "
                f"P90={self.prediction_intervals.get('p90_pct', 0.15):.2%}"
            )

        if LISTINGS_SLIM_PATH.exists():
            self.listings_lookup = pd.read_parquet(LISTINGS_SLIM_PATH).set_index("id")
            self.default_lat = float(pd.to_numeric(self.listings_lookup["latitude"], errors="coerce").median())
            self.default_lon = float(pd.to_numeric(self.listings_lookup["longitude"], errors="coerce").median())
            logger.info(f"Listings lookup loaded — {len(self.listings_lookup)} listings")

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
            row["dist_km_orla"] = float(_dist_to_nearest(lats, lons, ORLA_POINTS)[0])
            for category, pois in self.poi_coords.items():
                dist, count = poi_stats(lats, lons, pois)
                row[f"dist_km_{category}"] = float(dist[0])
                row[f"poi_{category}_500m"] = float(count[0])
        else:
            for name in FIXED_POIS:
                row[f"dist_km_{name}"] = 0.0
            row["dist_km_metro"] = 0.0
            row["dist_km_orla"] = 0.0
            for category in self.poi_coords:
                row[f"dist_km_{category}"] = 0.0
                row[f"poi_{category}_500m"] = 0.0
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

    def _bairro_features(self, neighbourhood: str) -> dict:
        """IPS do bairro; bairro desconhecido cai na mediana da cidade + flag."""
        stats = self.ips_lookup.get(normalize_bairro(neighbourhood))
        if stats:
            row = {f"bairro_{feat}": stats[feat] for feat in IPS_FEATURES}
            row["bairro_ips_missing"] = 0
        else:
            row = {f"bairro_{feat}": self.ips_medians.get(feat, 0.0) for feat in IPS_FEATURES}
            row["bairro_ips_missing"] = 1
        return row

    def _detect_event(self, d: date) -> Optional[str]:
        """reveillon | carnaval (±2 dias) | feriado (BR-RJ) | None."""
        if (d.month == 12 and d.day >= 27) or (d.month == 1 and d.day <= 2):
            return "reveillon"
        import holidays as holidays_lib
        br = holidays_lib.Brazil(subdiv="RJ", years=[d.year], categories=("public", "optional"))
        carnival_days = [dt for dt, name in br.items() if "Carnival" in name]
        if any(abs((d - c).days) <= 2 for c in carnival_days):
            return "carnaval"
        if d in br:
            return "feriado"
        return None

    def _seasonal_multiplier(self, target_date: Optional[date]) -> tuple[float, Optional[str]]:
        """Multiplicador de preço para a data e nota humana ("sábado de Carnaval — 1.16×")."""
        if not target_date or not self.seasonal_factors:
            return 1.0, None

        factors = self.seasonal_factors
        dow = target_date.weekday()
        idx = factors["dow"].get(dow, 1.0)

        event = self._detect_event(target_date)
        event_factor = factors["events"].get(event) if event else None
        if event_factor:
            idx *= event_factor

        mult = idx ** factors.get("dampening", 0.5)
        clip_low, clip_high = factors.get("clip", [0.8, 2.5])
        mult = round(max(clip_low, min(clip_high, mult)), 3)

        dow_names = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
        event_labels = {"reveillon": "Réveillon", "carnaval": "Carnaval", "feriado": "feriado"}
        note = dow_names[dow]
        if event and event_factor:
            note += f" de {event_labels[event]}"
        note += f" — ajuste {mult:.2f}×"
        return mult, note

    def _competition_features(self, neighbourhood: str, room_type: str,
                              comp_price_rank_override: Optional[float] = None) -> dict:
        """Lookup de estatísticas locais por bairro+tipo."""
        stats = self.competition_stats.get((neighbourhood, room_type), {})
        rank = comp_price_rank_override if comp_price_rank_override is not None else 0.5
        return {
            "comp_count": float(stats.get("count", 0)),
            "comp_price_p25": float(stats.get("p25", 0.0)),
            "comp_price_p50": float(stats.get("p50", 0.0)),
            "comp_price_p75": float(stats.get("p75", 0.0)),
            "comp_price_rank": rank,
        }

    def _build_features(self, req: PredictionRequest,
                        comp_price_rank_override: Optional[float] = None,
                        listing_id: Optional[int] = None) -> np.ndarray:
        row = {
            "accommodates": req.accommodates,
            "bathrooms": req.bathrooms,
            "bedrooms": req.bedrooms,
            "beds": req.beds,
            "minimum_nights": req.minimum_nights,
            "maximum_nights": req.maximum_nights,
            "calculated_host_listings_count": req.calculated_host_listings_count,
            "availability_365": req.availability_365,
            "host_is_superhost": int(req.host_is_superhost),
            "instant_bookable": int(req.instant_bookable),
            "has_availability": 1,
            # lat/lon cruas são features do modelo; sem coordenadas, mediana da cidade
            "latitude": req.latitude if req.latitude is not None else getattr(self, "default_lat", -22.97),
            "longitude": req.longitude if req.longitude is not None else getattr(self, "default_lon", -43.19),
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

        # Competição local (rank via override do two-pass ou 0.5 no primeiro pass)
        row.update(self._competition_features(req.neighbourhood, req.room_type,
                                              comp_price_rank_override))

        # Qualidade do bairro (IPS 2022)
        if self.ips_lookup:
            row.update(self._bairro_features(req.neighbourhood))

        # Imagens (CLIP) — scores reais da foto do listing quando disponíveis;
        # zeros para imóvel novo/sem foto (mesmo tratamento do fillna no treino)
        row.update(self._clip_features(listing_id))

        df = pd.DataFrame([row])

        # Garantir mesma ordem e conjunto de features do treino
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        return df

    CLIP_SCORE_COLS = ["luxury_score", "cleanliness_score", "brightness_score",
                       "professional_photo_score", "modern_style_score"]

    def _clip_features(self, listing_id: Optional[int]) -> dict:
        """Scores CLIP da foto real do listing; zeros sem foto (= fillna do treino)."""
        if (listing_id is not None and self.clip_lookup is not None
                and listing_id in self.clip_lookup.index):
            return self.clip_lookup.loc[listing_id].to_dict()
        cols = self.clip_lookup.columns if self.clip_lookup is not None else (
            self.CLIP_SCORE_COLS + [f"clip_emb_{i}" for i in range(20)]
        )
        return {c: 0.0 for c in cols}

    def _revenue_optimal(self, neighbourhood: str, room_type: str) -> tuple[Optional[float], Optional[float], str]:
        """Retorna (revenue_optimal_price, expected_occupancy, strategy) do segmento."""
        params = self.demand_params.get((neighbourhood, room_type))
        if not params or params.get("b") is None:
            return None, None, "fallback"

        a = params["a"]
        b = params["b"]
        comp_p50 = params["comp_p50"]
        comp_p75 = params.get("comp_p75", comp_p50 * 1.25)

        opt_price, opt_occ, strategy = _optimal_price(a, b, comp_p50, comp_p75)
        return round(opt_price, 2), round(opt_occ * 100, 1), strategy

    def predict_by_listing_id(self, listing_id: int, target_date: Optional[date] = None) -> ListingPredictionResponse:
        """Lookup listing real pelo id e retorna previsão com metadados do listing."""
        if self.listings_lookup is None or listing_id not in self.listings_lookup.index:
            raise KeyError(listing_id)

        row = self.listings_lookup.loc[listing_id]

        # Amenities: JSON string → list
        amenities_raw = row.get("amenities", "[]") or "[]"
        try:
            amenities = json.loads(amenities_raw) if isinstance(amenities_raw, str) else list(amenities_raw)
        except (json.JSONDecodeError, TypeError):
            amenities = []

        req = PredictionRequest(
            neighbourhood=str(row["neighbourhood_cleansed"]),
            room_type=str(row["room_type"]),
            accommodates=int(row["accommodates"]),
            bathrooms=float(row["bathrooms"] if pd.notna(row["bathrooms"]) else 1.0),
            bedrooms=int(row["bedrooms"] if pd.notna(row["bedrooms"]) else 0),
            beds=int(row["beds"] if pd.notna(row["beds"]) else 1),
            minimum_nights=int(row["minimum_nights"]),
            maximum_nights=int(row["maximum_nights"]),
            host_is_superhost=(str(row.get("host_is_superhost", "f")) == "t"),
            instant_bookable=(str(row.get("instant_bookable", "f")) == "t"),
            calculated_host_listings_count=int(row["calculated_host_listings_count"]),
            availability_365=int(row["availability_365"]),
            amenities=amenities,
            latitude=float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
            longitude=float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
            target_date=target_date,
        )

        base = self.predict(req, listing_id=listing_id)

        # Preço atual do dataset
        raw_price = str(row.get("price", "") or "")
        current_price = None
        try:
            current_price = float(raw_price.replace("$", "").replace(",", ""))
        except (ValueError, AttributeError):
            pass

        return ListingPredictionResponse(
            **base.model_dump(),
            listing_id=listing_id,
            listing_name=str(row.get("name", "")),
            listing_neighbourhood=str(row["neighbourhood_cleansed"]),
            listing_room_type=str(row["room_type"]),
            listing_accommodates=int(row["accommodates"]),
            listing_current_price=current_price,
            latitude=req.latitude,
            longitude=req.longitude,
        )

    def predict(self, req: PredictionRequest,
                listing_id: Optional[int] = None) -> PredictionResponse:
        # Pass 1: comp_price_rank = 0.5 (placeholder)
        X1 = self._build_features(req, listing_id=listing_id)
        log_pred1 = float(self.model.predict(X1)[0])
        market_price_1 = float(np.expm1(log_pred1))

        # Pass 2: recalcular comp_price_rank com base no preço previsto
        comp = self.competition_stats.get((req.neighbourhood, req.room_type), {})
        p25 = comp.get("p25", 0.0)
        p50 = comp.get("p50", 0.0)
        p75 = comp.get("p75", 0.0)

        if p25 > 0 and p50 > 0 and p75 > p25:
            actual_rank = _rank_from_percentiles(market_price_1, p25, p50, p75)
            X2 = self._build_features(req, comp_price_rank_override=actual_rank,
                                      listing_id=listing_id)
            log_pred2 = float(self.model.predict(X2)[0])
            market_price = float(np.expm1(log_pred2))
        else:
            market_price = market_price_1

        # Aviso se o preço previsto excede o P99 do treino (input suspeito)
        if self.price_p99 and market_price > self.price_p99 * 1.5:
            logger.warning(
                f"market_price R${market_price:.0f} excede P99 do treino "
                f"R${self.price_p99:.0f} × 1.5 — possível input outlier"
            )

        local_median = comp.get("p50")

        # Preço ótimo de receita via curva de demanda
        revenue_price, exp_occ, strategy = self._revenue_optimal(req.neighbourhood, req.room_type)

        # Intervalo de confiança via residuais OOF (P10/P90); fallback ±15%
        p10_pct = self.prediction_intervals.get("p10_pct", -0.15)
        p90_pct = self.prediction_intervals.get("p90_pct", 0.15)
        low = max(market_price * (1 + p10_pct), 1.0)
        high = market_price * (1 + p90_pct)

        # Ajuste sazonal por dia (camada pós-modelo: dow × evento, amortecido)
        seasonal_mult, seasonal_note = self._seasonal_multiplier(req.target_date)
        market_price *= seasonal_mult
        low *= seasonal_mult
        high *= seasonal_mult
        if revenue_price is not None:
            revenue_price = round(revenue_price * seasonal_mult, 2)

        return PredictionResponse(
            predicted_price=round(market_price, 2),
            price_range_low=round(low, 2),
            price_range_high=round(high, 2),
            local_median_price=round(local_median, 2) if local_median else None,
            seasonal_note=seasonal_note,
            seasonal_multiplier=seasonal_mult,
            revenue_optimal_price=revenue_price,
            expected_occupancy_pct=exp_occ,
            pricing_strategy=strategy,
        )
