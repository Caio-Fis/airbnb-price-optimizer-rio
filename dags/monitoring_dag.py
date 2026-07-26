"""
DAG: Monitoring
Detecta data drift (Evidently, Jun→Set) e quebra do modelo campeão.
Dispara retraining se necessário.

Ver a nota no topo de src/monitoring/drift.py: a segunda checagem NÃO mede
degradação de performance — não há conjunto rotulado fora da amostra. Ela pega
quebra grosseira (artefato corrompido, schema divergente).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
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
        from src.monitoring.drift import ModelSanityCheck
        report = ModelSanityCheck().run()
        context["ti"].xcom_push(key="model_broken", value=report["model_broken"])

    def decide_retraining(**context):
        ti = context["ti"]
        drift = ti.xcom_pull(task_ids="check_data_drift", key="drift_detected")
        broken = ti.xcom_pull(task_ids="check_model_performance", key="model_broken")
        should_retrain = bool(drift or broken)
        ti.xcom_push(key="should_retrain", value=should_retrain)
        return "trigger_retraining" if should_retrain else "skip_retraining"

    def skip_retraining(**context):
        print("No drift or degradation detected. Model is healthy.")

    t_drift = PythonOperator(task_id="check_data_drift", python_callable=check_data_drift)
    t_perf = PythonOperator(task_id="check_model_performance", python_callable=check_model_performance)
    # BranchPythonOperator, não PythonOperator: só o branch respeita o task_id
    # devolvido por decide_retraining. Com o operator comum, o retorno era
    # ignorado e trigger_retraining E skip_retraining rodavam os dois, todo dia —
    # o retreino disparava sempre, independentemente de drift.
    t_decide = BranchPythonOperator(
        task_id="decide_retraining", python_callable=decide_retraining
    )
    t_skip = PythonOperator(task_id="skip_retraining", python_callable=skip_retraining)
    t_trigger = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_dag",
        wait_for_completion=False,
    )

    [t_drift, t_perf] >> t_decide >> [t_trigger, t_skip]
