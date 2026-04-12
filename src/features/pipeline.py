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
    clip_path: str | None = None,
    yolo_path: str | None = None,
    reviews_path: str | None = None,
    competition_path: str | None = None,
    demand_path: str | None = None,
) -> str:
    logger.info("Merging all feature sources...")

    tabular = pd.read_parquet(tabular_path)
    seasonality = pd.read_parquet(seasonality_path)

    # Sazonalidade pelo last_scraped
    if "last_scraped" in tabular.columns:
        tabular["last_scraped"] = pd.to_datetime(tabular["last_scraped"])
        seasonality.index = pd.to_datetime(seasonality.index)
        tabular = tabular.merge(
            seasonality[["hw_seasonal", "hw_residual", "is_weekend", "month", "day_of_week"]],
            left_on="last_scraped",
            right_index=True,
            how="left",
        )

    # Imagem CLIP (opcional)
    clip_file = Path(clip_path) if clip_path else PROCESSED_DATA_PATH / "clip_features.parquet"
    if clip_file.exists() and "id" in tabular.columns:
        clip_feats = pd.read_parquet(clip_file)
        tabular = tabular.merge(clip_feats, left_on="id", right_on="listing_id", how="left")
        logger.info(f"CLIP features merged: {clip_feats.shape}")

    # Imagem YOLO (opcional)
    yolo_file = Path(yolo_path) if yolo_path else PROCESSED_DATA_PATH / "yolo_features.parquet"
    if yolo_file.exists() and "id" in tabular.columns:
        yolo_feats = pd.read_parquet(yolo_file)
        tabular = tabular.merge(yolo_feats, left_on="id", right_on="listing_id", how="left", suffixes=("", "_yolo"))
        logger.info(f"YOLO features merged: {yolo_feats.shape}")

    # Preencher NaN das features de imagem (listings sem foto)
    image_cols = [c for c in tabular.columns if c.startswith(("clip_", "yolo_", "has_", "luxury", "brightness"))]
    tabular[image_cols] = tabular[image_cols].fillna(0)

    # Merge com features geo (opcional — só se o arquivo existir)
    geo_file = PROCESSED_DATA_PATH / "geo_features.parquet"
    if geo_file.exists() and "id" in tabular.columns:
        geo_feats = pd.read_parquet(geo_file)
        tabular = tabular.merge(
            geo_feats,
            left_on="id",
            right_index=True,
            how="left",
        )
        geo_cols = [c for c in tabular.columns if c.startswith("dist_km_")]
        tabular[geo_cols] = tabular[geo_cols].fillna(tabular[geo_cols].median())
        logger.info(f"Geo features merged: {geo_feats.shape}")

    # Merge com features de competição local (opcional — só se o arquivo existir)
    if competition_path is None:
        competition_path = str(PROCESSED_DATA_PATH / "competition_features.parquet")
    competition_file = Path(competition_path)
    if competition_file.exists() and "id" in tabular.columns:
        competition_feats = pd.read_parquet(competition_file)
        tabular = tabular.merge(
            competition_feats,
            left_on="id",
            right_index=True,
            how="left",
        )
        comp_cols = [c for c in tabular.columns if c.startswith("comp_")]
        tabular[comp_cols] = tabular[comp_cols].fillna(tabular[comp_cols].median())
        logger.info(f"Competition features merged: {competition_feats.shape}")

    # Merge com features de demanda (opcional — só se o arquivo existir)
    demand_file = Path(demand_path) if demand_path else PROCESSED_DATA_PATH / "demand_features.parquet"
    if demand_file.exists() and "id" in tabular.columns:
        demand_feats = pd.read_parquet(demand_file)
        tabular = tabular.merge(
            demand_feats,
            left_on="id",
            right_index=True,
            how="left",
        )
        demand_cols = ["occupancy_rate", "price_premium", "revenue_optimal_price",
                       "expected_occupancy_at_optimal"]
        for col in demand_cols:
            if col in tabular.columns:
                tabular[col] = tabular[col].fillna(tabular[col].median())
        logger.info(f"Demand features merged: {demand_feats.shape}")

    output_path = PROCESSED_DATA_PATH / "final_features.parquet"
    tabular.to_parquet(output_path, index=False)
    logger.info(f"Final features saved to {output_path} — shape: {tabular.shape}")
    return str(output_path)
