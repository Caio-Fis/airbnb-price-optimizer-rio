"""
Feature engineering a partir de reviews:

- review_velocity       — reviews/mês (proxy de demanda)
- days_since_last_review — dias desde o último review (proxy de atividade)
- total_reviews          — volume histórico acumulado
- neg_keyword_ratio      — proporção de reviews com keywords negativas (proxy de problemas)
- n_neg_keywords         — contagem absoluta de reviews com keywords negativas

Estratégia para sinal negativo:
  Keyword pre-filter direto no corpus completo — sem modelo, sem sampling.
  Precision estimada em ~27% (calibrada na EDA do 02_reviews_analysis.ipynb),
  então neg_keyword_ratio superestima negativos reais por ~4x, mas é consistente
  entre listings e serve como feature ordinal.
"""
import os
import re
from pathlib import Path

import pandas as pd
from loguru import logger

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))

SNAPSHOT_DATE = pd.Timestamp("2025-09-26")

# Keywords calibradas na EDA (PT + EN + ES)
NEG_KEYWORDS = [
    # PT
    "ruim", "péssimo", "péssima", "horrível", "terrível", "lamentável",
    "problema", "problemas", "barulho", "barulhento", "sujo", "suja",
    "quebrado", "quebrada", "não funciona", "não funcionou", "decepcionante",
    "decepção", "insatisfeito", "reclamação", "descuidado", "mal cheiro",
    "barata", "baratas", "mosquito", "infestado", "falta", "faltou",
    # EN
    "dirty", "noisy", "broken", "terrible", "awful", "horrible", "disgusting",
    "cockroach", "bug", "mold", "mould", "smell", "smells", "misleading",
    "disappointing", "disappointed", "worst", "poor", "filthy", "unsafe",
    "didn't work", "not working", "false", "lied", "scam",
    # ES
    "sucio", "roto", "horrible", "pésimo", "ruido", "ruidoso", "decepcionante",
]
# Compilado uma vez, case-insensitive via lowercase antecipado no texto
_NEG_PATTERN = re.compile("|".join(re.escape(k) for k in NEG_KEYWORDS))


class ReviewFeaturePipeline:
    def run(self) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

        reviews = pd.read_parquet(
            PROCESSED_DATA_PATH / "reviews_set2025.parquet",
            columns=["listing_id", "id", "date", "comments"],
        )
        logger.info(f"Reviews carregados: {len(reviews):,} linhas")

        features = self._velocity_recency(reviews)
        neg_signal = self._neg_signal(reviews)
        features = features.join(neg_signal, how="left")

        # Listings sem nenhum match de keyword ficam com 0
        features["n_neg_keywords"] = features["n_neg_keywords"].fillna(0).astype(int)
        features["neg_keyword_ratio"] = features["neg_keyword_ratio"].fillna(0.0)

        output_path = PROCESSED_DATA_PATH / "reviews_features.parquet"
        features.to_parquet(output_path)
        logger.info(
            f"Reviews features salvas em {output_path} — "
            f"{len(features):,} listings, {len(features.columns)} features"
        )
        return str(output_path)

    # ------------------------------------------------------------------
    def _velocity_recency(self, reviews: pd.DataFrame) -> pd.DataFrame:
        stats = reviews.groupby("listing_id").agg(
            first_review=("date", "min"),
            last_review=("date", "max"),
            total_reviews=("id", "count"),
        )

        months_active = (
            (SNAPSHOT_DATE - stats["first_review"]).dt.days / 30.44
        ).clip(lower=1)

        stats["review_velocity"] = (stats["total_reviews"] / months_active).round(4)
        stats["days_since_last_review"] = (
            SNAPSHOT_DATE - stats["last_review"]
        ).dt.days

        return stats[["review_velocity", "days_since_last_review", "total_reviews"]]

    def _neg_signal(self, reviews: pd.DataFrame) -> pd.DataFrame:
        has_text = reviews.dropna(subset=["comments"]).copy()

        # lowercase uma vez → evita flag case=False por linha
        is_neg = has_text["comments"].str.lower().str.contains(
            _NEG_PATTERN, regex=True, na=False
        )

        neg_counts = (
            has_text[is_neg]
            .groupby("listing_id")
            .size()
            .rename("n_neg_keywords")
        )
        total_counts = (
            has_text.groupby("listing_id")
            .size()
            .rename("n_reviews_with_text")
        )

        df = pd.concat([neg_counts, total_counts], axis=1).fillna(0)
        df["neg_keyword_ratio"] = (
            df["n_neg_keywords"] / df["n_reviews_with_text"]
        ).round(4)

        return df[["n_neg_keywords", "neg_keyword_ratio"]]
