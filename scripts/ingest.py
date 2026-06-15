import pandas as pd
import sqlalchemy
import os
import redis
import pickle

MARIADB_HOST = os.getenv("MARIADB_HOST", "127.0.0.1")
DB_URL = f"mysql+pymysql://mlops_user:mlops_pass@{MARIADB_HOST}:3306/creditcard_db"
XLS_PATH = os.getenv("XLS_PATH", "/home/blond/mlops/data/default of credit card clients.xls")

print("Reading dataset...")
df = pd.read_excel(XLS_PATH, header=1)
print(f"Shape: {df.shape}")
df.rename(columns={"default payment next month": "default_payment"}, inplace=True)
df.columns = df.columns.str.strip()

print("Connecting to MariaDB ColumnStore...")
engine = sqlalchemy.create_engine(DB_URL)

print("Loading Star Schema tables...")
dim_client = df[["ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"]].copy()
dim_client.rename(columns={"ID": "client_id"}, inplace=True)
dim_client.to_sql("dim_client", con=engine, if_exists="append", index=False)
print(f"dim_client loaded: {len(dim_client)} rows")

dim_repayment = df[["ID", "PAY_0", "PAY_2", "PAY_3", "PAY_AMT1", "PAY_AMT2"]].copy()
dim_repayment.rename(columns={"ID": "repayment_id"}, inplace=True)
dim_repayment.to_sql("dim_repayment", con=engine, if_exists="append", index=False)
print(f"dim_repayment loaded: {len(dim_repayment)} rows")

fact = df[["ID", "LIMIT_BAL", "BILL_AMT1", "BILL_AMT2", "default_payment"]].copy()
fact.insert(1, "repayment_id", df["ID"])
fact.rename(columns={"ID": "client_id"}, inplace=True)
fact.to_sql("fact_credit_default", con=engine, if_exists="append", index=False)
print(f"fact_credit_default loaded: {len(fact)} rows")

print("Caching raw data to Redis...")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
r = redis.Redis(host=REDIS_HOST, port=6379)
r.set("raw_data", pickle.dumps(df))
print("Data cached to Redis successfully.")
print("Ingestion complete.")