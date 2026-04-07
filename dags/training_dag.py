"""
DAG: Training
Treina XGBoost e LightGBM, registra no MLflow, promove o melhor modelo.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    "owner": "airbnb-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="training_dag",
    description="Train, evaluate and register price prediction model",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["training"],
) as dag:

    wait_for_features = ExternalTaskSensor(
        task_id="wait_for_features",
        external_dag_id="feature_engineering_dag",
        external_task_id="merge_features",
        timeout=3600,
        mode="reschedule",
    )

    def train_xgboost(**context):
        from src.training.train import train_model
        run_id = train_model(model_type="xgboost")
        context["ti"].xcom_push(key="xgb_run_id", value=run_id)

    def train_lightgbm(**context):
        from src.training.train import train_model
        run_id = train_model(model_type="lightgbm")
        context["ti"].xcom_push(key="lgb_run_id", value=run_id)

    def evaluate_and_promote(**context):
        from src.training.evaluate import select_best_model
        ti = context["ti"]
        best_run_id = select_best_model(
            run_ids=[
                ti.xcom_pull(task_ids="train_xgboost", key="xgb_run_id"),
                ti.xcom_pull(task_ids="train_lightgbm", key="lgb_run_id"),
            ]
        )
        context["ti"].xcom_push(key="best_run_id", value=best_run_id)

    def register_model(**context):
        from src.training.evaluate import register_champion
        best_run_id = context["ti"].xcom_pull(
            task_ids="evaluate_and_promote", key="best_run_id"
        )
        register_champion(best_run_id)

    t_xgb = PythonOperator(task_id="train_xgboost", python_callable=train_xgboost)
    t_lgb = PythonOperator(task_id="train_lightgbm", python_callable=train_lightgbm)
    t_eval = PythonOperator(task_id="evaluate_and_promote", python_callable=evaluate_and_promote)
    t_reg = PythonOperator(task_id="register_model", python_callable=register_model)

    wait_for_features >> [t_xgb, t_lgb] >> t_eval >> t_reg
