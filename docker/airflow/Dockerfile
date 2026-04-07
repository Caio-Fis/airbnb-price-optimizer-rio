FROM apache/airflow:2.9.1-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Instalar pacotes Airflow extra
RUN pip install --no-cache-dir \
    apache-airflow-providers-google==10.19.0 \
    apache-airflow-providers-common-sql==1.14.0

COPY src/ /opt/airflow/src/
