"""
Features de qualidade do bairro — IPS 2022 (Índice de Progresso Social, IPP/data.rio).

Fonte: data/external/ips_bairros_2022.csv (159 bairros oficiais), colunas:
  ips_2022        — índice geral (0-100)
  seguranca_2022  — dimensão Segurança Pessoal
  renda_2010      — renda domiciliar per capita (Censo 2010)

Matching de nomes: normalização (minúsculas, sem acento) + overrides manuais em
data/external/bairro_name_map.json para divergências Inside Airbnb ↔ IPP.
"""
import json
import os
import unicodedata
from pathlib import Path

import pandas as pd
from loguru import logger

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
EXTERNAL_DATA_PATH = Path(os.getenv("EXTERNAL_DATA_PATH", "data/external"))
IPS_CSV = EXTERNAL_DATA_PATH / "ips_bairros_2022.csv"
NAME_MAP_JSON = EXTERNAL_DATA_PATH / "bairro_name_map.json"

IPS_FEATURES = ["ips_2022", "seguranca_2022", "renda_2010"]


def normalize_bairro(name: str) -> str:
    """minúsculas, sem acentos, espaços colapsados."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(name.lower().split())


def load_ips_lookup() -> dict[str, dict]:
    """bairro normalizado → {ips_2022, seguranca_2022, renda_2010}.

    Aplica os overrides de nome (Inside Airbnb → IPP) se o JSON existir.
    """
    df = pd.read_csv(IPS_CSV)
    lookup = {
        normalize_bairro(row["bairro"]): {f: float(row[f]) for f in IPS_FEATURES}
        for _, row in df.iterrows()
        if pd.notna(row["ips_2022"])
    }

    if NAME_MAP_JSON.exists():
        with open(NAME_MAP_JSON) as f:
            overrides = json.load(f)
        for airbnb_name, ipp_name in overrides.items():
            key_ipp = normalize_bairro(ipp_name)
            if key_ipp in lookup:
                lookup[normalize_bairro(airbnb_name)] = lookup[key_ipp]
            else:
                logger.warning(f"Override aponta para bairro IPP inexistente: {ipp_name}")

    return lookup


class BairroFeaturePipeline:
    """Gera bairro_features.parquet indexado por listing id."""

    def run(self) -> str:
        listings = pd.read_parquet(
            PROCESSED_DATA_PATH / "listings_set2025.parquet",
            columns=["id", "neighbourhood_cleansed"],
        )
        lookup = load_ips_lookup()

        keys = listings["neighbourhood_cleansed"].map(normalize_bairro)
        features = pd.DataFrame(index=listings["id"])
        for feat in IPS_FEATURES:
            features[f"bairro_{feat}"] = keys.map(
                lambda k: lookup.get(k, {}).get(feat)
            ).values
        features["bairro_ips_missing"] = features["bairro_ips_2022"].isna().astype(int)

        matched = int((features["bairro_ips_missing"] == 0).sum())
        logger.info(
            f"IPS matching: {matched:,}/{len(features):,} listings "
            f"({100 * matched / len(features):.1f}%)"
        )

        output_path = PROCESSED_DATA_PATH / "bairro_features.parquet"
        features.to_parquet(output_path)
        logger.info(f"Bairro features salvas em {output_path}")
        return str(output_path)
