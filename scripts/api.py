import pickle
import redis
import mlflow
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime

app = FastAPI(title="Credit Card Default Prediction API")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# --- PostgreSQL connection ---
def get_pg_conn():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", 5432),
        dbname=os.getenv("PG_DB", "mlops_logs"),
        user=os.getenv("PG_USER", "mlops_user"),
        password=os.getenv("PG_PASSWORD", "mlops_pass")
    )

def init_prediction_log_table():
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_log (
                id            SERIAL PRIMARY KEY,
                timestamp     TIMESTAMP NOT NULL,
                limit_bal     FLOAT,
                age           INT,
                pay_0         INT,
                pay_2         INT,
                pay_3         INT,
                bill_amt1     FLOAT,
                bill_amt2     FLOAT,
                pay_amt1      FLOAT,
                pay_amt2      FLOAT,
                prediction    INT,
                probability   FLOAT,
                risk          VARCHAR(10)
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Prediction log table ready.")
    except Exception as e:
        print(f"PostgreSQL init error: {e}")

def log_prediction(data: dict, prediction: int, probability: float, risk: str):
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO prediction_log
                (timestamp, limit_bal, age, pay_0, pay_2, pay_3,
                 bill_amt1, bill_amt2, pay_amt1, pay_amt2,
                 prediction, probability, risk)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            datetime.utcnow(),
            data["LIMIT_BAL"], data["AGE"], data["PAY_0"],
            data["PAY_2"], data["PAY_3"], data["BILL_AMT1"],
            data["BILL_AMT2"], data["PAY_AMT1"], data["PAY_AMT2"],
            prediction, probability, risk
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Prediction logging error: {e}")

# --- Load model and scaler from Redis ---
model = None
scaler = None
model_loading_error = None

try:
    r = redis.Redis(host=REDIS_HOST, port=6379)
    model = pickle.loads(r.get("best_model"))
    scaler = pickle.loads(r.get("scaler"))
    print("Model and scaler loaded from Redis successfully!")
except Exception as e:
    model_loading_error = str(e)
    print(f"Error loading model: {e}")

# --- Init DB table on startup ---
init_prediction_log_table()

# --- Request Schema ---
class ClientData(BaseModel):
    LIMIT_BAL: float
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    BILL_AMT1: float
    BILL_AMT2: float
    PAY_AMT1: float
    PAY_AMT2: float

# --- Endpoints ---
@app.get("/")
def root():
    return {
        "message": "Credit Card Default Prediction API",
        "model_status": "loaded" if model is not None else "not_loaded",
        "error": model_loading_error if model_loading_error else None
    }

@app.post("/predict")
def predict(data: ClientData):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not available: {model_loading_error}")
    df = pd.DataFrame([data.dict()])
    try:
        scaled = scaler.transform(df)
        prediction = int(model.predict(scaled)[0])
        probability = round(float(model.predict_proba(scaled)[0][1]), 4)
        risk = "HIGH" if prediction == 1 else "LOW"
        log_prediction(data.dict(), prediction, probability, risk)
        return {
            "prediction": prediction,
            "probability_of_default": probability,
            "risk": risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/status")
def status():
    return {
        "status": "available" if model is not None else "unavailable",
        "error": model_loading_error if model_loading_error else None
    }