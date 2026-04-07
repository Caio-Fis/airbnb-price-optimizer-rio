"""
DAG: Ingestion
Baixa dados do Inside Airbnb e imagens dos listings para o GCS.
Roda semanalmente para capturar novos snapshots.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airbnb-ml",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ingestion_dag",
    description="Download Airbnb data and images from Inside Airbnb",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion"],
) as dag:

    def download_listings(**context):
        from src.ingestion.download_data import download_listings_csv
        path = download_listings_csv()
        context["ti"].xcom_push(key="listings_path", value=path)

    def download_calendar(**context):
        from src.ingestion.download_data import download_calendar_csv
        path = download_calendar_csv()
        context["ti"].xcom_push(key="calendar_path", value=path)

    def download_images(**context):
        from src.ingestion.download_data import download_listing_images
        listings_path = context["ti"].xcom_pull(
            task_ids="download_listings", key="listings_path"
        )
        download_listing_images(listings_path, max_images=5000)

    def upload_to_gcs(**context):
        from src.ingestion.download_data import upload_raw_to_gcs
        upload_raw_to_gcs()

    t1 = PythonOperator(task_id="download_listings", python_callable=download_listings)
    t2 = PythonOperator(task_id="download_calendar", python_callable=download_calendar)
    t3 = PythonOperator(task_id="download_images", python_callable=download_images)
    t4 = PythonOperator(task_id="upload_to_gcs", python_callable=upload_to_gcs)

    [t1, t2] >> t3 >> t4
