from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import redis

default_args = {
    "owner": "saurav",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

def check_recall_threshold(**context):
    """Pull latest recall from Redis. Retrain only if below 0.65."""
    r = redis.Redis(host="mlops_redis", port=6379)
    val = r.get("latest_recall")
    if val is None:
        print("No recall found — triggering retrain.")
        return "model_training"
    recall = float(val.decode())
    print(f"Latest recall: {recall}")
    if recall < 0.65:
        print("Recall below threshold — retraining.")
        return "model_training"
    print("Recall acceptable — skipping retrain.")
    return "skip_training"

with DAG(
    dag_id="mlops_pipeline",
    default_args=default_args,
    description="MLOps pipeline for credit card default prediction",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    ingest = BashOperator(
        task_id="data_ingestion",
        bash_command="cd /opt/airflow && python3 scripts/ingest.py",
    )
    preprocess = BashOperator(
        task_id="data_preprocessing",
        bash_command="cd /opt/airflow && python3 scripts/preprocess.py",
    )
    train = BashOperator(
        task_id="model_training",
        bash_command="cd /opt/airflow && python3 scripts/train.py",
    )
    monitor = BashOperator(
        task_id="model_monitoring",
        bash_command="cd /opt/airflow && python3 scripts/monitor.py",
    )
    check_recall = BranchPythonOperator(
        task_id="check_recall_threshold",
        python_callable=check_recall_threshold,
    )
    retrain = BashOperator(
        task_id="model_retraining",
        bash_command="cd /opt/airflow && python3 scripts/train.py",
    )
    skip_training = EmptyOperator(
        task_id="skip_training",
    )
    ingest >> preprocess >> train >> monitor >> check_recall >> [retrain, skip_training]










    