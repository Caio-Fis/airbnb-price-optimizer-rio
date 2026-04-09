"""
DAG: Ingestion
Baixa dados de listings e calendário do Airbnb para o GCS.

Fluxo:
  1. check_snapshots — verifica se Inside Airbnb publicou novo snapshot
  2a. [novo snapshot oficial] → download_official_snapshot
  2b. [sem atualização] → scrape_airbnb → cria snapshot próprio com Playwright
  3. download_images — baixa até 5000 fotos dos listings
  4. upload_to_gcs — sobe tudo para o GCS

Cadência: @monthly (não diária — precisamos de variação temporal para a curva de demanda).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

KNOWN_SNAPSHOTS = ["2025-06-24", "2025-09-26"]

default_args = {
    "owner": "airbnb-ml",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ingestion_dag",
    description="Download or scrape Airbnb data and upload to GCS",
    schedule="@monthly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion"],
) as dag:

    def check_snapshots(**context):
        """Verifica Inside Airbnb. Se novo snapshot → rota oficial; senão → scraper."""
        from src.ingestion.airbnb_scraper import check_inside_airbnb

        new_dates = check_inside_airbnb(known_dates=KNOWN_SNAPSHOTS)
        context["ti"].xcom_push(key="new_snapshot_dates", value=new_dates)

        if new_dates:
            return "download_official_snapshot"
        return "scrape_airbnb"

    def download_official_snapshot(**context):
        """Baixa novo snapshot do Inside Airbnb (quando disponível)."""
        from src.ingestion.download_data import download_listings_csv, download_calendar_csv

        new_dates = context["ti"].xcom_pull(task_ids="check_snapshots", key="new_snapshot_dates")
        for snapshot_date in (new_dates or []):
            listings_path = download_listings_csv(snapshot_date=snapshot_date)
            download_calendar_csv(snapshot_date=snapshot_date)
            context["ti"].xcom_push(key="listings_path", value=listings_path)

    def scrape_airbnb(**context):
        """Cria snapshot próprio via scraper do Airbnb (Playwright)."""
        from src.ingestion.airbnb_scraper import scrape_rio_listings
        from pathlib import Path
        import os

        output_dir = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
        listings_path, _ = scrape_rio_listings(output_dir=output_dir)
        context["ti"].xcom_push(key="listings_path", value=str(listings_path))

    def download_images(**context):
        from src.ingestion.download_data import download_listing_images

        # Tentar obter listings_path de qualquer um dos dois caminhos anteriores
        listings_path = (
            context["ti"].xcom_pull(task_ids="download_official_snapshot", key="listings_path")
            or context["ti"].xcom_pull(task_ids="scrape_airbnb", key="listings_path")
        )
        if listings_path:
            download_listing_images(listings_path, max_images=5000)

    def upload_to_gcs(**context):
        from src.ingestion.download_data import upload_raw_to_gcs
        upload_raw_to_gcs()

    t0 = BranchPythonOperator(task_id="check_snapshots", python_callable=check_snapshots)
    t_official = PythonOperator(task_id="download_official_snapshot",
                                python_callable=download_official_snapshot)
    t_scrape = PythonOperator(task_id="scrape_airbnb", python_callable=scrape_airbnb)
    t3 = PythonOperator(task_id="download_images", python_callable=download_images,
                        trigger_rule="none_failed_min_one_success")
    t4 = PythonOperator(task_id="upload_to_gcs", python_callable=upload_to_gcs,
                        trigger_rule="none_failed_min_one_success")

    t0 >> [t_official, t_scrape] >> t3 >> t4
