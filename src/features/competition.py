"""
Features de mercado local: competição por bairro × tipo de quarto.

Para cada listing:
  comp_count           — n° de concorrentes no mesmo bairro+tipo
  comp_price_p25/p50/p75 — distribuição de preços locais
  comp_price_rank      — percentil do listing dentro do grupo

Também salva competition_stats.joblib para uso em inferência.
"""
import os
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))


class CompetitionFeaturePipeline:
    def run(self) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(
            PROCESSED_DATA_PATH / "tabular_features.parquet",
            columns=["id", "price", "neighbourhood_cleansed", "room_type"],
        )
        df = df.dropna(subset=["price", "neighbourhood_cleansed", "room_type"])

        # Estatísticas por grupo
        group_stats = (
            df.groupby(["neighbourhood_cleansed", "room_type"])["price"]
            .agg(
                comp_count="count",
                comp_price_p25=lambda x: x.quantile(0.25),
                comp_price_p50="median",
                comp_price_p75=lambda x: x.quantile(0.75),
            )
            .reset_index()
        )

        df = df.merge(group_stats, on=["neighbourhood_cleansed", "room_type"], how="left")

        # Percentil de cada listing dentro do seu grupo (0=mais barato, 1=mais caro)
        df["comp_price_rank"] = (
            df.groupby(["neighbourhood_cleansed", "room_type"])["price"]
            .rank(pct=True)
        )

        feature_cols = [
            "id", "comp_count",
            "comp_price_p25", "comp_price_p50", "comp_price_p75",
            "comp_price_rank",
        ]
        features = df[feature_cols].set_index("id")

        output_path = PROCESSED_DATA_PATH / "competition_features.parquet"
        features.to_parquet(output_path)
        logger.info(f"Competition features saved to {output_path} — shape: {features.shape}")

        # Lookup para inferência: (neighbourhood, room_type) → stats
        stats_dict = {}
        for _, row in group_stats.iterrows():
            key = (row["neighbourhood_cleansed"], row["room_type"])
            stats_dict[key] = {
                "count": int(row["comp_count"]),
                "p25": float(row["comp_price_p25"]),
                "p50": float(row["comp_price_p50"]),
                "p75": float(row["comp_price_p75"]),
            }

        stats_path = PROCESSED_DATA_PATH / "competition_stats.joblib"
        joblib.dump(stats_dict, stats_path)
        logger.info(f"Competition stats saved to {stats_path} — {len(stats_dict)} groups")

        return str(output_path)
