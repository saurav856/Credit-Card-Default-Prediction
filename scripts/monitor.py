import pandas as pd
import pickle
import redis
import sqlalchemy
import psycopg2
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

MARIADB_HOST = os.getenv("MARIADB_HOST", "127.0.0.1")
DB_URL = f"mysql+pymysql://mlops_user:mlops_pass@{MARIADB_HOST}:3306/creditcard_db"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
FEATURES = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3",
            "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]
REPORTS_DIR = os.getenv("REPORTS_DIR", "/home/blond/mlops/data/reports")

print("Loading reference data from MariaDB ColumnStore...")
engine = sqlalchemy.create_engine(DB_URL)
df = pd.read_sql("""
    SELECT f.LIMIT_BAL, f.BILL_AMT1, f.BILL_AMT2, f.default_payment,
           c.AGE,
           r.PAY_0, r.PAY_2, r.PAY_3, r.PAY_AMT1, r.PAY_AMT2
    FROM fact_credit_default f
    JOIN dim_client c ON f.client_id = c.client_id
    JOIN dim_repayment r ON f.repayment_id = r.repayment_id
""", con=engine)
df.rename(columns={"default_payment": "target"}, inplace=True)
reference = df.sample(n=1000, random_state=42)

print("Loading live prediction data from PostgreSQL...")
pg_conn = psycopg2.connect(
    host=os.getenv("PG_HOST", "localhost"),
    port=os.getenv("PG_PORT", 5432),
    dbname=os.getenv("PG_DB", "mlops_logs"),
    user=os.getenv("PG_USER", "mlops_user"),
    password=os.getenv("PG_PASSWORD", "mlops_pass")
)
current = pd.read_sql("""
    SELECT limit_bal AS "LIMIT_BAL", age AS "AGE",
           pay_0 AS "PAY_0", pay_2 AS "PAY_2", pay_3 AS "PAY_3",
           bill_amt1 AS "BILL_AMT1", bill_amt2 AS "BILL_AMT2",
           pay_amt1 AS "PAY_AMT1", pay_amt2 AS "PAY_AMT2",
           prediction
    FROM prediction_log
    ORDER BY timestamp DESC
    LIMIT 500
""", con=pg_conn)
pg_conn.close()

if len(current) < 10:
    print("Not enough live data — falling back to MariaDB sample.")
    current = df.sample(n=500, random_state=99)
    current["prediction"] = None

print("Loading model and scaler from Redis...")
r = redis.Redis(host=REDIS_HOST, port=6379)
model = pickle.loads(r.get("best_model"))
scaler = pickle.loads(r.get("scaler"))

reference["prediction"] = model.predict(scaler.transform(reference[FEATURES]))

if "prediction" not in current.columns or current["prediction"].isnull().all():
    current["prediction"] = model.predict(scaler.transform(current[FEATURES]))

if "target" not in reference.columns:
    reference["target"] = df.sample(n=1000, random_state=42)["target"].values
if "target" not in current.columns:
    current["target"] = 0

ref_df = reference[FEATURES + ["target", "prediction"]].reset_index(drop=True)
cur_df = current[FEATURES + ["target", "prediction"]].reset_index(drop=True)

os.makedirs(REPORTS_DIR, exist_ok=True)

print("Running drift report...")
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=ref_df, current_data=cur_df)
drift_report.save_html(f"{REPORTS_DIR}/drift_report.html")
print("Drift report saved.")

print("Running classification performance report...")
performance_report = Report(metrics=[ClassificationPreset()])
performance_report.run(reference_data=ref_df, current_data=cur_df)
performance_report.save_html(f"{REPORTS_DIR}/performance_report.html")
print("Performance report saved.")

print("Running combined dashboard...")
dashboard = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
dashboard.run(reference_data=ref_df, current_data=cur_df)
dashboard.save_html(f"{REPORTS_DIR}/monitoring_dashboard.html")
print("Monitoring dashboard saved.")

print("Extracting live recall and pushing to Redis...")
try:
    result = performance_report.as_dict()
    metrics = result["metrics"]
    live_recall = None
    for m in metrics:
        if "recall" in str(m).lower():
            try:
                live_recall = m["result"]["current"]["recall"]
                break
            except (KeyError, TypeError):
                continue
    if live_recall is not None and live_recall > 0.0:
        r.set("live_recall", str(live_recall))
        print(f"Live recall pushed to Redis: {live_recall:.4f}")
    else:
        print("Live recall = 0 (no real targets) — falling back to latest_recall.")
        fallback = r.get("latest_recall")
        if fallback:
            r.set("live_recall", fallback)
            print(f"Fallback recall pushed to Redis: {float(fallback):.4f}")
except Exception as e:
    print(f"Recall extraction failed: {e} — using latest_recall as fallback.")
    fallback = r.get("latest_recall")
    if fallback:
        r.set("live_recall", fallback)
        print(f"Fallback recall pushed to Redis: {float(fallback):.4f}")