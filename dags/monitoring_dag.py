"""
DAG: Monitoring
Detecta data drift e performance degradation usando Evidently AI.
Dispara retraining se necessário.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "airbnb-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="monitoring_dag",
    description="Monitor data drift and trigger retraining if needed",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["monitoring"],
) as dag:

    def check_data_drift(**context):
        from src.monitoring.drift import DriftDetector
        detector = DriftDetector()
        report = detector.run()
        drift_detected = report["dataset_drift"]
        context["ti"].xcom_push(key="drift_detected", value=drift_detected)
        context["ti"].xcom_push(key="drift_share", value=report["drift_share"])

    def check_model_performance(**context):
        from src.monitoring.drift import PerformanceMonitor
        monitor = PerformanceMonitor()
        report = monitor.run()
        degraded = report["rmse_degraded"]
        context["ti"].xcom_push(key="performance_degraded", value=degraded)

    def decide_retraining(**context):
        ti = context["ti"]
        drift = ti.xcom_pull(task_ids="check_data_drift", key="drift_detected")
        degraded = ti.xcom_pull(task_ids="check_model_performance", key="performance_degraded")
        should_retrain = drift or degraded
        context["ti"].xcom_push(key="should_retrain", value=should_retrain)
        return "trigger_retraining" if should_retrain else "skip_retraining"

    def skip_retraining(**context):
        print("No drift or degradation detected. Model is healthy.")

    t_drift = PythonOperator(task_id="check_data_drift", python_callable=check_data_drift)
    t_perf = PythonOperator(task_id="check_model_performance", python_callable=check_model_performance)
    t_decide = PythonOperator(task_id="decide_retraining", python_callable=decide_retraining)
    t_skip = PythonOperator(task_id="skip_retraining", python_callable=skip_retraining)
    t_trigger = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_dag",
        wait_for_completion=False,
    )

    [t_drift, t_perf] >> t_decide >> [t_trigger, t_skip]
