"""
Ingestion: baixa listings, calendário e imagens do Inside Airbnb.

Snapshots disponíveis para Rio de Janeiro:
- 2025-09-26 (mais recente) → usado para treino
- 2025-06-24 (anterior)     → usado para histórico/Holt-Winters
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests
from google.cloud import storage
from loguru import logger

RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
IMAGES_PATH = Path(os.getenv("IMAGES_PATH", "data/images"))
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "airbnb-price-optimizer")
CITY = os.getenv("INSIDE_AIRBNB_CITY", "rio-de-janeiro")

INSIDE_AIRBNB_BASE = "https://data.insideairbnb.com/brazil"

# Snapshots ordenados do mais antigo para o mais recente
CITY_SNAPSHOTS = {
    "rio-de-janeiro": [
        f"{INSIDE_AIRBNB_BASE}/rj/rio-de-janeiro/2025-06-24/data",
        f"{INSIDE_AIRBNB_BASE}/rj/rio-de-janeiro/2025-09-26/data",  # mais recente = treino
    ],
}

SNAPSHOT_DATES = {
    "rio-de-janeiro": ["2025-06-24", "2025-09-26"],
}


def _download_file(url: str, output_path: Path, timeout: int = 300) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url}")
    response = requests.get(
        url,
        stream=True,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
    logger.info(f"Saved to {output_path}")


def download_listings_csv(city: str = CITY) -> str:
    """Baixa o snapshot mais recente de listings (usado para treino)."""
    latest_url = CITY_SNAPSHOTS[city][-1]
    date = SNAPSHOT_DATES[city][-1]
    output_path = RAW_DATA_PATH / f"{city}_listings_{date}.csv.gz"
    # Symlink sem data para compatibilidade com o resto do pipeline
    symlink = RAW_DATA_PATH / f"{city}_listings.csv.gz"

    _download_file(f"{latest_url}/listings.csv.gz", output_path)

    symlink.unlink(missing_ok=True)
    symlink.symlink_to(output_path.name)
    return str(output_path)


def download_calendar_csv(city: str = CITY) -> str:
    """
    Baixa todos os snapshots de calendário e concatena em um único arquivo.
    Isso fornece uma série histórica longa para o Holt-Winters.
    """
    snapshots = CITY_SNAPSHOTS[city]
    dates = SNAPSHOT_DATES[city]
    dfs = []

    for url, date in zip(snapshots, dates):
        output_path = RAW_DATA_PATH / f"{city}_calendar_{date}.csv.gz"
        if not output_path.exists():
            _download_file(f"{url}/calendar.csv.gz", output_path)

        df = pd.read_csv(output_path, compression="gzip", parse_dates=["date"])
        df["snapshot_date"] = date
        dfs.append(df)
        logger.info(f"Loaded calendar snapshot {date}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    # Remover duplicatas (mesma data + listing no mesmo snapshot)
    combined = combined.drop_duplicates(subset=["listing_id", "date"])
    combined = combined.sort_values("date")

    output_path = RAW_DATA_PATH / f"{city}_calendar.csv.gz"
    combined.to_csv(output_path, index=False, compression="gzip")
    logger.info(f"Combined calendar: {len(combined)} rows → {output_path}")
    return str(output_path)


def download_listing_images(listings_path: str, max_images: int = 5000) -> None:
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(listings_path, compression="gzip", usecols=["id", "picture_url"])
    df = df.dropna(subset=["picture_url"]).head(max_images)

    logger.info(f"Downloading {len(df)} images...")
    failed = 0
    for _, row in df.iterrows():
        img_path = IMAGES_PATH / f"{row['id']}.jpg"
        if img_path.exists():
            continue
        try:
            resp = requests.get(
                row["picture_url"],
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            img_path.write_bytes(resp.content)
            time.sleep(0.05)
        except Exception as e:
            logger.warning(f"Failed to download image {row['id']}: {e}")
            failed += 1

    logger.info(f"Images downloaded. Failed: {failed}/{len(df)}")


def upload_raw_to_gcs() -> None:
    client = storage.Client()
    bucket = client.bucket(GCP_BUCKET_NAME)

    for folder in [RAW_DATA_PATH, IMAGES_PATH]:
        for file_path in Path(folder).rglob("*"):
            if file_path.is_file() and not file_path.is_symlink():
                blob_name = f"raw/{file_path.relative_to(Path('data'))}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                logger.info(f"Uploaded {file_path} → gs://{GCP_BUCKET_NAME}/{blob_name}")
