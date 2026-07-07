"""
Feature engineering geoespacial:

POIs hard-coded (landmarks icônicos — interpretáveis no modelo):
  dist_km_praia_ipanema, dist_km_praia_copacabana, dist_km_cristo,
  dist_km_aeroporto_galeao, dist_km_aeroporto_sdu, dist_km_centro

POIs via OSM (buscados uma vez e cacheados em JSON — o mesmo cache alimenta
o serving, garantindo paridade treino/inferência):
  dist_km_metro — estação de metrô mais próxima
  dist_km_{categoria} + poi_{categoria}_500m — vizinhança comercial/lazer
  (bares, restaurantes, cafés/padarias, academias, parques, mercados,
  atrações turísticas, vida noturna)

Distâncias em km via Haversine (BallTree para os POIs de vizinhança).
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

# Pontos ao longo da orla carioca (baía de Guanabara + praias oceânicas)
# Cobertura: Flamengo → Botafogo → Urca → Leme → Copacabana → Ipanema →
#            Leblon → Vidigal → São Conrado → Barra → Recreio
ORLA_POINTS: list[tuple[float, float]] = [
    (-22.9320, -43.1748),  # Flamengo norte (baía)
    (-22.9420, -43.1780),  # Flamengo sul (baía)
    (-22.9500, -43.1800),  # Botafogo (baía)
    (-22.9494, -43.1656),  # Urca (baía)
    (-22.9592, -43.1672),  # Leme norte
    (-22.9650, -43.1718),  # Leme
    (-22.9711, -43.1822),  # Copacabana central
    (-22.9760, -43.1920),  # Copacabana sul
    (-22.9876, -43.1976),  # Arpoador
    (-22.9847, -43.2034),  # Ipanema leste
    (-22.9862, -43.2150),  # Ipanema central
    (-22.9870, -43.2245),  # Ipanema/Leblon
    (-22.9870, -43.2340),  # Leblon
    (-22.9950, -43.2470),  # Leblon/Vidigal
    (-23.0000, -43.2640),  # São Conrado
    (-23.0073, -43.3650),  # Barra norte
    (-23.0200, -43.4200),  # Barra central
    (-23.0250, -43.4700),  # Recreio
    (-23.0200, -43.5000),  # Recreio oeste
]

# Categorias de POI de vizinhança (tags OSM) — impacto local no preço
POI_CATEGORIES: dict[str, dict] = {
    "bar":         {"amenity": ["bar", "pub"]},
    "restaurant":  {"amenity": "restaurant"},
    "cafe_bakery": {"amenity": "cafe", "shop": "bakery"},
    "gym":         {"leisure": "fitness_centre"},
    "park":        {"leisure": "park"},
    "supermarket": {"shop": "supermarket"},
    "attraction":  {"tourism": ["attraction", "museum", "viewpoint"]},
    "nightclub":   {"amenity": "nightclub"},
}

# Cache do OSM para não consultar a API toda vez
_OSM_CACHE = PROCESSED_DATA_PATH / "geo_metro_cache.json"


def poi_cache_path(category: str) -> Path:
    return PROCESSED_DATA_PATH / f"geo_poi_{category}.json"


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


def _fetch_pois(name: str, tags: dict, cache_path: Path) -> list[tuple[float, float]]:
    """Busca POIs do Rio via OSM por tags. Resultado cacheado em JSON (lat, lon)."""
    if cache_path.exists():
        logger.info(f"Cache de {name} encontrado — pulando OSM.")
        with open(cache_path) as f:
            return [tuple(p) for p in json.load(f)]

    logger.info(f"Buscando {name} via OSM (tags={tags})...")
    import osmnx as ox
    ox.settings.requests_timeout = 180
    gdf = ox.features_from_place("Rio de Janeiro, Brazil", tags=tags)
    # Centroide de cada geometria → (lat, lon)
    gdf = gdf.to_crs(epsg=4326)
    pois = [(round(geom.centroid.y, 6), round(geom.centroid.x, 6)) for geom in gdf.geometry]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(pois, f)
    logger.info(f"{len(pois)} POIs de {name} encontrados e cacheados.")
    return pois


def _fetch_metro_stations() -> list[tuple[float, float]]:
    return _fetch_pois("metro", {"station": "subway"}, _OSM_CACHE)


def _dist_to_nearest(lats: np.ndarray, lons: np.ndarray,
                     stations: list[tuple[float, float]]) -> np.ndarray:
    """Distância ao ponto mais próximo de uma lista."""
    dists = np.stack(
        [_haversine(lats, lons, lat, lon) for lat, lon in stations],
        axis=1,
    )
    return dists.min(axis=1)


EARTH_RADIUS_KM = 6371.0


def poi_stats(lats: np.ndarray, lons: np.ndarray,
              pois: list[tuple[float, float]],
              radius_km: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    (distância km ao POI mais próximo, contagem de POIs em radius_km).

    BallTree haversine: escala para 43k listings × milhares de POIs.
    Usada tanto no pipeline offline quanto no serving — paridade por construção.
    """
    from sklearn.neighbors import BallTree

    tree = BallTree(np.radians(np.asarray(pois)), metric="haversine")
    points = np.radians(np.column_stack([lats, lons]))
    dist_rad, _ = tree.query(points, k=1)
    counts = tree.query_radius(points, r=radius_km / EARTH_RADIUS_KM, count_only=True)
    return dist_rad[:, 0] * EARTH_RADIUS_KM, counts.astype(float)


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

        # Orla mais próxima (baía + praias oceânicas)
        features["dist_km_orla"] = _dist_to_nearest(lats, lons, ORLA_POINTS).round(3)
        logger.info("Distância à orla calculada.")

        # Vizinhança comercial/lazer: distância + densidade em 500m por categoria
        for category, tags in POI_CATEGORIES.items():
            try:
                pois = _fetch_pois(category, tags, poi_cache_path(category))
                if not pois:
                    continue
                dist, count = poi_stats(lats, lons, pois)
                features[f"dist_km_{category}"] = dist.round(3)
                features[f"poi_{category}_500m"] = count
                logger.info(f"POIs de {category}: {len(pois)} pontos processados.")
            except Exception as e:
                logger.warning(f"OSM falhou para {category} — features omitidas. Erro: {e}")

        output_path = PROCESSED_DATA_PATH / "geo_features.parquet"
        features.to_parquet(output_path)
        logger.info(
            f"Geo features salvas em {output_path} — "
            f"{len(features):,} listings, {len(features.columns)} features"
        )
        return str(output_path)
