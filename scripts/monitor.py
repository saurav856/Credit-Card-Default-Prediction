import pandas as pd
import pickle
import sqlalchemy
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping
import os

DB_URL = "mysql+pymysql://mlops_user:mlops_pass@127.0.0.1:3306/creditcard_db"

print("Loading reference data...")
engine = sqlalchemy.create_engine(DB_URL)
df = pd.read_sql("SELECT * FROM credit_card_default", con=engine)
df.drop(columns=["ID"], inplace=True)
df.rename(columns={"default_payment": "target"}, inplace=True)

reference = df.sample(n=5000, random_state=42)
current = df.sample(n=1000, random_state=99)

model = pickle.load(open("models/best_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

feature_cols = [c for c in df.columns if c != "target"]

ref_scaled = scaler.transform(reference[feature_cols])
cur_scaled = scaler.transform(current[feature_cols])

reference["prediction"] = model.predict(ref_scaled)
current["prediction"] = model.predict(cur_scaled)

column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=feature_cols
)

report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

os.makedirs("logs", exist_ok=True)
report.save_html("logs/monitoring_report.html")
print("Monitoring report saved to logs/monitoring_report.html")