"""
Feature engineering geoespacial:

POIs hard-coded (landmarks icônicos — interpretáveis no modelo):
  dist_km_praia_ipanema, dist_km_praia_copacabana, dist_km_cristo,
  dist_km_aeroporto_galeao, dist_km_aeroporto_sdu, dist_km_centro

POIs via OSM (infra de mobilidade — buscados uma vez e cacheados):
  dist_km_metro_mais_proximo

Distâncias em km via fórmula de Haversine (vetorizada, sem dependência de geo libs).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))

# ---------------------------------------------------------------------------
# POIs fixos: (lat, lon)
# ---------------------------------------------------------------------------
FIXED_POIS = {
    "praia_ipanema":       (-22.9838, -43.2096),
    "praia_copacabana":    (-22.9711, -43.1822),
    "praia_barra_tijuca":  (-23.0073, -43.3650),
    "cristo_redentor":     (-22.9519, -43.2105),
    "aeroporto_galeao":    (-22.8090, -43.2436),
    "aeroporto_sdu":       (-22.9105, -43.1631),
    "centro_rj":           (-22.9068, -43.1729),
    "lapa":                (-22.9147, -43.1800),
    "maracana":            (-22.9122, -43.2302),
}

# Cache do OSM para não consultar a API toda vez
_OSM_CACHE = PROCESSED_DATA_PATH / "geo_metro_cache.json"


def _haversine(lat1: np.ndarray, lon1: np.ndarray,
               lat2: float, lon2: float) -> np.ndarray:
    """Distância em km entre arrays de pontos e um ponto fixo."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def _fetch_metro_stations() -> list[tuple[float, float]]:
    """Busca estações de metrô do Rio via OSM. Resultado cacheado em JSON."""
    if _OSM_CACHE.exists():
        logger.info("Metro cache encontrado — pulando OSM.")
        with open(_OSM_CACHE) as f:
            return [tuple(p) for p in json.load(f)]

    logger.info("Buscando estações de metrô via OSM...")
    import osmnx as ox
    gdf = ox.features_from_place(
        "Rio de Janeiro, Brazil",
        tags={"station": "subway"},
    )
    # Centroide de cada geometria → (lat, lon)
    gdf = gdf.to_crs(epsg=4326)
    stations = [(geom.centroid.y, geom.centroid.x) for geom in gdf.geometry]

    _OSM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_OSM_CACHE, "w") as f:
        json.dump(stations, f)
    logger.info(f"{len(stations)} estações de metrô encontradas e cacheadas.")
    return stations


def _dist_to_nearest(lats: np.ndarray, lons: np.ndarray,
                     stations: list[tuple[float, float]]) -> np.ndarray:
    """Distância ao ponto mais próximo de uma lista."""
    dists = np.stack(
        [_haversine(lats, lons, lat, lon) for lat, lon in stations],
        axis=1,
    )
    return dists.min(axis=1)


class GeoFeaturePipeline:
    def run(self) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

        listings = pd.read_parquet(
            PROCESSED_DATA_PATH / "listings_set2025.parquet",
            columns=["id", "latitude", "longitude"],
        )
        listings["latitude"] = pd.to_numeric(listings["latitude"], errors="coerce")
        listings["longitude"] = pd.to_numeric(listings["longitude"], errors="coerce")
        listings = listings.dropna(subset=["latitude", "longitude"])

        lats = listings["latitude"].to_numpy()
        lons = listings["longitude"].to_numpy()

        features = pd.DataFrame(index=listings["id"])

        # POIs fixos
        for name, (poi_lat, poi_lon) in FIXED_POIS.items():
            features[f"dist_km_{name}"] = _haversine(lats, lons, poi_lat, poi_lon).round(3)
        logger.info(f"POIs fixos calculados: {len(FIXED_POIS)}")

        # Metrô mais próximo (OSM)
        try:
            stations = _fetch_metro_stations()
            if stations:
                features["dist_km_metro"] = _dist_to_nearest(lats, lons, stations).round(3)
                logger.info("Distância ao metrô calculada.")
        except Exception as e:
            logger.warning(f"OSM falhou — dist_km_metro omitida. Erro: {e}")

        output_path = PROCESSED_DATA_PATH / "geo_features.parquet"
        features.to_parquet(output_path)
        logger.info(
            f"Geo features salvas em {output_path} — "
            f"{len(features):,} listings, {len(features.columns)} features"
        )
        return str(output_path)
