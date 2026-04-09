"""
DAG: Feature Engineering
Processa features tabulares (encoding, Holt-Winters) e de imagens (CLIP, YOLO).
Depende do ingestion_dag via dataset sensor.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    "owner": "airbnb-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="feature_engineering_dag",
    description="Build tabular and image features",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["features"],
) as dag:

    wait_for_ingestion = ExternalTaskSensor(
        task_id="wait_for_ingestion",
        external_dag_id="ingestion_dag",
        external_task_id="upload_to_gcs",
        timeout=3600,
        mode="reschedule",
    )

    def build_tabular_features(**context):
        from src.features.tabular import TabularFeaturePipeline
        pipeline = TabularFeaturePipeline()
        output_path = pipeline.run(fit=True)
        context["ti"].xcom_push(key="tabular_path", value=output_path)

    def build_seasonality_features(**context):
        from src.features.tabular import SeasonalityPipeline
        pipeline = SeasonalityPipeline()
        output_path = pipeline.run()
        context["ti"].xcom_push(key="seasonality_path", value=output_path)

    def build_clip_features(**context):
        from src.features.image_clip import CLIPFeatureExtractor
        extractor = CLIPFeatureExtractor()
        output_path = extractor.run()
        context["ti"].xcom_push(key="clip_path", value=output_path)

    def build_yolo_features(**context):
        from src.features.image_yolo import YOLOFeatureExtractor
        extractor = YOLOFeatureExtractor()
        output_path = extractor.run()
        context["ti"].xcom_push(key="yolo_path", value=output_path)

    def build_competition_features(**context):
        from src.features.competition import CompetitionFeaturePipeline
        output_path = CompetitionFeaturePipeline().run()
        context["ti"].xcom_push(key="competition_path", value=output_path)

    def build_demand_features(**context):
        from src.features.demand import DemandPipeline
        output_path = DemandPipeline().run()
        context["ti"].xcom_push(key="demand_path", value=output_path)

    def merge_features(**context):
        from src.features.pipeline import merge_all_features
        ti = context["ti"]
        output_path = merge_all_features(
            tabular_path=ti.xcom_pull(task_ids="build_tabular_features", key="tabular_path"),
            seasonality_path=ti.xcom_pull(task_ids="build_seasonality_features", key="seasonality_path"),
            clip_path=ti.xcom_pull(task_ids="build_clip_features", key="clip_path"),
            yolo_path=ti.xcom_pull(task_ids="build_yolo_features", key="yolo_path"),
            competition_path=ti.xcom_pull(task_ids="build_competition_features", key="competition_path"),
            demand_path=ti.xcom_pull(task_ids="build_demand_features", key="demand_path"),
        )
        context["ti"].xcom_push(key="features_path", value=output_path)

    t_tabular = PythonOperator(task_id="build_tabular_features", python_callable=build_tabular_features)
    t_seasonal = PythonOperator(task_id="build_seasonality_features", python_callable=build_seasonality_features)
    t_clip = PythonOperator(task_id="build_clip_features", python_callable=build_clip_features)
    t_yolo = PythonOperator(task_id="build_yolo_features", python_callable=build_yolo_features)
    t_competition = PythonOperator(task_id="build_competition_features", python_callable=build_competition_features)
    t_demand = PythonOperator(task_id="build_demand_features", python_callable=build_demand_features)
    t_merge = PythonOperator(task_id="merge_features", python_callable=merge_features)

    # competition depende do tabular (precisa de preços por bairro)
    # demand depende do tabular + competition
    wait_for_ingestion >> [t_tabular, t_seasonal, t_clip, t_yolo]
    t_tabular >> t_competition >> t_demand
    [t_tabular, t_seasonal, t_clip, t_yolo, t_competition, t_demand] >> t_merge
