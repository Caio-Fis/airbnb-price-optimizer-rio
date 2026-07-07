"""
Gera o GeoJSON comprimido consumido pelo mapa do frontend.

Lê listings_slim.parquet e produz data/processed/listings_map.json.gz:
FeatureCollection com id como STRING (ids > 2^53 corrompem em JavaScript)
e propriedades mínimas para tooltip/card (nome, bairro, tipo, preço).
"""
import gzip
import json
import os
from pathlib import Path

import pandas as pd
from loguru import logger

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
INPUT_PATH = PROCESSED_DATA_PATH / "listings_slim.parquet"
OUTPUT_PATH = PROCESSED_DATA_PATH / "listings_map.json.gz"


def build_map_listings() -> str:
    df = pd.read_parquet(
        INPUT_PATH,
        columns=["id", "name", "neighbourhood_cleansed", "room_type",
                 "accommodates", "latitude", "longitude", "price"],
    )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    price = (
        df["price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["price_num"] = pd.to_numeric(price, errors="coerce")

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 5), round(lat, 5)],
            },
            "properties": {
                "id": str(id_),          # string: ids > 2^53 corrompem em JS
                "name": name,
                "nb": nb,
                "rt": rt,
                "acc": int(acc),
                "price": round(p, 2) if pd.notna(p) else None,
            },
        }
        for id_, name, nb, rt, acc, lat, lon, p in zip(
            df["id"], df["name"], df["neighbourhood_cleansed"], df["room_type"],
            df["accommodates"], df["latitude"], df["longitude"], df["price_num"],
        )
    ]

    geojson = {"type": "FeatureCollection", "features": features}
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    logger.info(f"{len(features):,} listings → {OUTPUT_PATH} ({size_mb:.1f} MB gzip)")
    return str(OUTPUT_PATH)


if __name__ == "__main__":
    build_map_listings()
