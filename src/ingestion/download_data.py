"""
Ingestion: baixa listings, calendário e imagens do Inside Airbnb.
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

# Inside Airbnb base URL
INSIDE_AIRBNB_BASE = "https://data.insideairbnb.com/brazil"

CITY_URLS = {
    "rio-de-janeiro": f"{INSIDE_AIRBNB_BASE}/rj/rio-de-janeiro/2024-03-23/data",
}


def download_listings_csv(city: str = CITY) -> str:
    base_url = CITY_URLS[city]
    url = f"{base_url}/listings.csv.gz"
    output_path = RAW_DATA_PATH / f"{city}_listings.csv.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading listings from {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Listings saved to {output_path}")
    return str(output_path)


def download_calendar_csv(city: str = CITY) -> str:
    base_url = CITY_URLS[city]
    url = f"{base_url}/calendar.csv.gz"
    output_path = RAW_DATA_PATH / f"{city}_calendar.csv.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading calendar from {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Calendar saved to {output_path}")
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
            resp = requests.get(row["picture_url"], timeout=10)
            resp.raise_for_status()
            img_path.write_bytes(resp.content)
            time.sleep(0.05)  # rate limiting
        except Exception as e:
            logger.warning(f"Failed to download image {row['id']}: {e}")
            failed += 1

    logger.info(f"Images downloaded. Failed: {failed}/{len(df)}")


def upload_raw_to_gcs() -> None:
    client = storage.Client()
    bucket = client.bucket(GCP_BUCKET_NAME)

    for folder in [RAW_DATA_PATH, IMAGES_PATH]:
        for file_path in Path(folder).rglob("*"):
            if file_path.is_file():
                blob_name = f"raw/{file_path.relative_to(Path('data'))}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                logger.info(f"Uploaded {file_path} → gs://{GCP_BUCKET_NAME}/{blob_name}")
