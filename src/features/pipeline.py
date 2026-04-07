"""
Feature pipeline: merge de todas as fontes de features em um dataset final.
"""
import os
from pathlib import Path

import pandas as pd
from loguru import logger

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))


def merge_all_features(
    tabular_path: str,
    seasonality_path: str,
    clip_path: str,
    yolo_path: str,
) -> str:
    logger.info("Merging all feature sources...")

    tabular = pd.read_parquet(tabular_path)
    seasonality = pd.read_parquet(seasonality_path)
    clip_feats = pd.read_parquet(clip_path)
    yolo_feats = pd.read_parquet(yolo_path)

    # Adicionar features de sazonalidade pelo scrape_date ou last_scraped
    if "last_scraped" in tabular.columns:
        tabular["last_scraped"] = pd.to_datetime(tabular["last_scraped"])
        seasonality.index = pd.to_datetime(seasonality.index)
        tabular = tabular.merge(
            seasonality[["hw_seasonal", "hw_residual", "is_weekend", "month", "day_of_week"]],
            left_on="last_scraped",
            right_index=True,
            how="left",
        )

    # Merge com features de imagem pelo listing_id
    if "id" in tabular.columns:
        tabular = tabular.merge(clip_feats, left_on="id", right_on="listing_id", how="left")
        tabular = tabular.merge(yolo_feats, left_on="id", right_on="listing_id", how="left", suffixes=("", "_yolo"))

    # Preencher NaN das features de imagem (listings sem foto)
    image_cols = [c for c in tabular.columns if c.startswith(("clip_", "yolo_", "has_", "luxury", "brightness"))]
    tabular[image_cols] = tabular[image_cols].fillna(0)

    output_path = PROCESSED_DATA_PATH / "final_features.parquet"
    tabular.to_parquet(output_path, index=False)
    logger.info(f"Final features saved to {output_path} — shape: {tabular.shape}")
    return str(output_path)
