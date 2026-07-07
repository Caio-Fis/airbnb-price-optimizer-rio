"""
Relatório (read-only) do matching entre neighbourhood_cleansed (Inside Airbnb)
e os nomes de bairro do IPS (data.rio). Uso: python scripts/check_bairro_match.py
"""
import pandas as pd

from src.features.bairro import load_ips_lookup, normalize_bairro

listings = pd.read_parquet(
    "data/processed/listings_set2025.parquet", columns=["neighbourhood_cleansed"]
)
counts = listings["neighbourhood_cleansed"].value_counts()
lookup = load_ips_lookup()

unmatched = []
for bairro, n in counts.items():
    if normalize_bairro(bairro) not in lookup:
        unmatched.append((bairro, n))

total = counts.sum()
missing = sum(n for _, n in unmatched)
print(f"{len(counts)} bairros no Inside Airbnb, {len(unmatched)} sem match no IPS")
print(f"Listings sem IPS: {missing:,} de {total:,} ({100 * missing / total:.1f}%)\n")
for bairro, n in unmatched:
    print(f"  {bairro:40s} {n:6,} listings")
