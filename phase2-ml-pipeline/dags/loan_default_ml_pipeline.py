"""End-to-end ML pipeline DAG: medallion datamart -> train -> inference -> monitor."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

PYTHON = "cd /opt/airflow && PYTHONPATH=/opt/airflow python"

# When Trigger DAG is clicked without a logical date, Airflow uses today (2026) which is
# outside our data window. Fall back to a known-good snapshot so tasks still run.
SNAPSHOT = (
    "{% if ds >= '2023-01-01' and ds <= '2024-12-01' %}{{ ds }}"
    "{% else %}2024-07-01{% endif %}"
)

with DAG(
    dag_id="loan_default_ml_pipeline",
    default_args=default_args,
    description="Medallion datamart, train, score, and monitor loan default models",
    schedule_interval="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["loan-default", "ml-pipeline"],
) as dag:
    start = EmptyOperator(task_id="start")

    run_medallion = BashOperator(
        task_id="run_medallion_snapshot",
        bash_command=f'{PYTHON} utils/datamart/run_snapshot.py --snapshotdate "{SNAPSHOT}"',
    )

    build_training_table = BashOperator(
        task_id="build_training_table",
        bash_command=f'{PYTHON} utils/datamart/build_training_table.py',
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            "{% set snap = ds if ds >= '2023-01-01' and ds <= '2024-12-01' else '2024-07-01' %}"
            "{% set snap_month = macros.datetime.strptime(snap, '%Y-%m-%d').month %}"
            "{% if snap_month in [1, 4, 7, 10] %}"
            f'{PYTHON} utils/ml/train.py --snapshotdate "{SNAPSHOT}" --force-retrain'
            "{% else %}"
            f'{PYTHON} utils/ml/train.py --snapshotdate "{SNAPSHOT}"'
            "{% endif %}"
        ),
    )

    backfill_inference = BashOperator(
        task_id="backfill_inference",
        bash_command=f'{PYTHON} utils/ml/backfill_inference.py --snapshotdate "{SNAPSHOT}"',
    )

    run_inference = BashOperator(
        task_id="run_inference",
        bash_command=f'{PYTHON} utils/ml/inference.py --snapshotdate "{SNAPSHOT}"',
    )

    run_monitoring = BashOperator(
        task_id="run_monitoring",
        bash_command=f'{PYTHON} utils/ml/monitor.py --snapshotdate "{SNAPSHOT}"',
    )

    end = EmptyOperator(task_id="end")

    start >> run_medallion >> build_training_table >> train_model
    train_model >> backfill_inference >> run_inference >> run_monitoring >> end
