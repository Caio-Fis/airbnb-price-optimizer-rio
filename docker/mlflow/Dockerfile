FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    psycopg2-binary==2.9.9 \
    google-cloud-storage==2.16.0 \
    boto3==1.34.0

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "${MLFLOW_BACKEND_STORE_URI}", \
     "--default-artifact-root", "${MLFLOW_ARTIFACT_ROOT}", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
