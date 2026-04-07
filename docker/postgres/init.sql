-- Cria banco separado para MLflow (Airflow usa o banco padrão 'airflow')
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;
