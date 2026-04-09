"""
Converte os arquivos CSV.gz brutos para Parquet.

Listings:    todas as colunas, compressão snappy.
Calendários: apenas listing_id/date/available (price é 100% nulo nesse scrape).
Reviews:     todas as colunas, parse de datas.

Uso:
    python scripts/convert_to_parquet.py
    make parquet
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(exist_ok=True)

LISTINGS = {
    "listings_jun2025": "rio-de-janeiro_listings_2025-06-24.csv.gz",
    "listings_set2025": "rio-de-janeiro_listings_2025-09-26.csv.gz",
}

CALENDARS = {
    "calendar_jun2025": "rio-de-janeiro_calendar_2025-06-24.csv.gz",
    "calendar_set2025": "rio-de-janeiro_calendar_2025-09-26.csv.gz",
}

REVIEWS = {
    "reviews_jun2025": "rio-de-janeiro_reviews_2025-06-24.csv.gz",
    "reviews_set2025": "rio-de-janeiro_reviews_2025-09-26.csv.gz",
}


def convert_listings(name: str, filename: str) -> None:
    out = PROCESSED / f"{name}.parquet"
    if out.exists():
        print(f"  {out.name} já existe — pulando.")
        return
    print(f"  Convertendo {filename}...")
    df = pd.read_csv(RAW / filename, compression="gzip", low_memory=False)
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  -> {out.name} ({out.stat().st_size / 1e6:.1f} MB, {len(df):,} linhas)")


def convert_calendar(name: str, filename: str) -> None:
    out = PROCESSED / f"{name}.parquet"
    if out.exists():
        print(f"  {out.name} já existe — pulando.")
        return
    print(f"  Convertendo {filename}  (arquivo grande, aguarde)...")
    df = pd.read_csv(
        RAW / filename,
        compression="gzip",
        parse_dates=["date"],
        usecols=["listing_id", "date", "available"],
    )
    df["available"] = df["available"].astype("category")
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  -> {out.name} ({out.stat().st_size / 1e6:.1f} MB, {len(df):,} linhas)")


def convert_reviews(name: str, filename: str) -> None:
    out = PROCESSED / f"{name}.parquet"
    src = RAW / filename
    if out.exists():
        print(f"  {out.name} já existe — pulando.")
        return
    if not src.exists():
        print(f"  {filename} não encontrado em data/raw/ — pulando.")
        return
    print(f"  Convertendo {filename}...")
    df = pd.read_csv(src, compression="gzip", parse_dates=["date"])
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  -> {out.name} ({out.stat().st_size / 1e6:.1f} MB, {len(df):,} linhas)")


if __name__ == "__main__":
    print("=== Listings ===")
    for name, f in LISTINGS.items():
        convert_listings(name, f)

    print("\n=== Calendários ===")
    for name, f in CALENDARS.items():
        convert_calendar(name, f)

    print("\n=== Reviews ===")
    for name, f in REVIEWS.items():
        convert_reviews(name, f)

    print("\nPronto.")
