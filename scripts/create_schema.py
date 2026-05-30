import sqlalchemy

DB_URL = "mysql+pymysql://mlops_user:mlops_pass@127.0.0.1:3306/creditcard_db"
engine = sqlalchemy.create_engine(DB_URL)

sql = """
DROP TABLE IF EXISTS fact_credit_default;
DROP TABLE IF EXISTS dim_client;
DROP TABLE IF EXISTS dim_repayment;

CREATE TABLE dim_client (
    client_id INT PRIMARY KEY,
    SEX INT,
    EDUCATION INT,
    MARRIAGE INT,
    AGE INT
);

CREATE TABLE dim_repayment (
    repayment_id INT PRIMARY KEY,
    PAY_0 INT,
    PAY_AMT1 FLOAT,
    PAY_AMT2 FLOAT
);

CREATE TABLE fact_credit_default (
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    client_id INT,
    repayment_id INT,
    LIMIT_BAL FLOAT,
    BILL_AMT1 FLOAT,
    default_payment INT,
    FOREIGN KEY (client_id) REFERENCES dim_client(client_id),
    FOREIGN KEY (repayment_id) REFERENCES dim_repayment(repayment_id)
);
"""

with engine.connect() as conn:
    for statement in sql.strip().split(";"):
        s = statement.strip()
        if s:
            conn.execute(sqlalchemy.text(s))
    conn.commit()

print("Star schema created successfully.")