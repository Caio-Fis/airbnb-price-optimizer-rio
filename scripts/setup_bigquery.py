"""
Setup BigQuery external tables for Looker Studio dashboard.

Uploads relevant Parquet files to GCS and creates a BigQuery dataset
with external tables pointing to them.

Usage:
    python scripts/setup_bigquery.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Config
PROJECT_ID = "project-b0fa2d28-630e-4922-b41"
BUCKET = "airbnb-price-optimizer"
GCS_PREFIX = "looker"
DATASET = "airbnb_rj"
LOCATION = "US"

PROCESSED = Path("data/processed")

# Files to upload and their BigQuery table names
TABLES = {
    "final_features": "final_features.parquet",
    "listings_jun2025": "listings_jun2025.parquet",
    "listings_set2025": "listings_set2025.parquet",
    "calendar_jun2025": "calendar_jun2025.parquet",
    "calendar_set2025": "calendar_set2025.parquet",
}


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def upload_to_gcs():
    print("\n[1/3] Uploading Parquet files to GCS...")
    for table_name, filename in TABLES.items():
        local_path = PROCESSED / filename
        gcs_path = f"gs://{BUCKET}/{GCS_PREFIX}/{filename}"
        if not local_path.exists():
            print(f"  SKIP {filename} — not found locally")
            continue
        run(f"gcloud storage cp {local_path} {gcs_path}")
        print(f"  OK  {filename} → {gcs_path}")


def create_dataset():
    print(f"\n[2/3] Creating BigQuery dataset '{DATASET}'...")
    result = run(
        f"bq --project_id={PROJECT_ID} ls --datasets",
        check=False,
    )
    if DATASET in result.stdout:
        print(f"  Dataset '{DATASET}' already exists, skipping.")
        return
    run(
        f"bq --project_id={PROJECT_ID} mk "
        f"--dataset --location={LOCATION} {PROJECT_ID}:{DATASET}"
    )
    print(f"  OK  Dataset {DATASET} created.")


def create_external_tables():
    print("\n[3/3] Creating external tables in BigQuery...")

    ext_def = {"sourceFormat": "PARQUET", "autodetect": True}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        def_path = f.name

    for table_name, filename in TABLES.items():
        gcs_uri = f"gs://{BUCKET}/{GCS_PREFIX}/{filename}"
        full_table = f"{PROJECT_ID}:{DATASET}.{table_name}"

        ext_def["sourceUris"] = [gcs_uri]
        with open(def_path, "w") as f:
            json.dump(ext_def, f)

        run(f"bq --project_id={PROJECT_ID} rm -f --table {full_table}", check=False)
        run(
            f"bq --project_id={PROJECT_ID} mk "
            f"--external_table_definition={def_path} "
            f"{full_table}"
        )
        print(f"  OK  {table_name} → {gcs_uri}")


def print_summary():
    print("\n" + "=" * 60)
    print("Done! Connect Looker Studio:")
    print(f"  Project:  {PROJECT_ID}")
    print(f"  Dataset:  {DATASET}")
    print(f"  Tables:   {', '.join(TABLES.keys())}")
    print("\nIn Looker Studio:")
    print("  1. New Report → Add data → BigQuery")
    print(f"  2. Project: {PROJECT_ID}")
    print(f"  3. Dataset: {DATASET}")
    print("  4. Start with table: final_features")
    print("=" * 60)


if __name__ == "__main__":
    upload_to_gcs()
    create_dataset()
    create_external_tables()
    print_summary()
