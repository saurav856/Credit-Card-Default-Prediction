import sqlalchemy

DB_URL = "mysql+pymysql://mlops_user:mlops_pass@127.0.0.1:3306/creditcard_db"
engine = sqlalchemy.create_engine(DB_URL)

sql_statements = [
    "DROP TABLE IF EXISTS fact_credit_default",
    "DROP TABLE IF EXISTS dim_client", 
    "DROP TABLE IF EXISTS dim_repayment",
    """CREATE TABLE dim_client (
        client_id INT,
        SEX INT,
        EDUCATION INT,
        MARRIAGE INT,
        AGE INT
    ) ENGINE=Columnstore""",
    """CREATE TABLE dim_repayment (
        repayment_id INT,
        PAY_0 INT,
        PAY_AMT1 FLOAT,
        PAY_AMT2 FLOAT
    ) ENGINE=Columnstore""",
    """CREATE TABLE fact_credit_default (
        client_id INT,
        repayment_id INT,
        LIMIT_BAL FLOAT,
        BILL_AMT1 FLOAT,
        default_payment INT
    ) ENGINE=Columnstore"""
]

with engine.connect() as conn:
    for statement in sql_statements:
        conn.execute(sqlalchemy.text(statement))
    conn.commit()

print("Star schema created with ColumnStore engine successfully.")
