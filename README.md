# Credit Card Default Prediction — MLOps Pipeline

End-to-end MLOps pipeline for automated credit card default prediction.

**Live demo:** https://credit-card-prediction.streamlit.app

## Dataset
- **Primary:** UCI Default of Credit Card Clients (30,000 rows, 25 features)  
- **Source:** Yeh & Lien (2009), UCI Machine Learning Repository

## Pipeline Architecture

| Stage | Description |
|-------|-------------|
| 1 | Data ingestion: XLS → ETL → Great Expectations validation → MariaDB ColumnStore (Star Schema) |
| 2 | Preprocessing: feature selection → train/test split → SMOTE → StandardScaler → Redis cache |
| 3 | Training: Logistic Regression + Random Forest → MLflow tracking → Model Registry |
| 4 | Serving: FastAPI `/predict` endpoint (local) + standalone Streamlit dashboard (loads model directly for public deployment) |
| 5 | Monitoring: Evidently drift reports → BranchPythonOperator → automated Airflow retrain trigger |

## Tech Stack
- **Orchestration:** Apache Airflow 2.10.0 (Docker)
- **Data Warehouse:** MariaDB ColumnStore 23.02.4 (Star Schema)
- **Operational Store:** PostgreSQL (prediction logging)
- **Cache:** Redis 7
- **ML:** scikit-learn, SMOTE (imbalanced-learn), MLflow
- **Validation:** Great Expectations 1.4.4
- **Monitoring:** Evidently 0.4.30
- **API:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Infrastructure:** Docker + Docker Compose, Ubuntu 24.04

## Model Results

| Model | Recall (Class 1) | AUC-ROC |
|-------|-----------------|---------|
| Logistic Regression | 0.7717 | 0.7089 |
| Random Forest | 0.4763 | 0.7229 |

**Production model:** Logistic Regression (threshold=0.40, optimised for recall — false negatives carry higher business cost in credit risk).

## Project Structure

```
├── dags/          # Airflow DAG
├── data/          
├── docker/        # Docker Compose config
├── models/        # Trained model + scaler (used by standalone Streamlit app)
├── scripts/       # Pipeline scripts
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. Clone repo
2. Copy `.env.example` to `.env` and fill in secrets
3. Start containers:
```bash
cd docker
docker compose up -d
docker exec mlops_mariadb provision mcs1
```
4. Start Airflow: `docker start mlops_airflow` → localhost:8080
5. Start MLflow: `mlflow ui --host 0.0.0.0 --port 5000`
6. Start API: `uvicorn api:app --reload`
7. Start dashboard: `streamlit run app.py` (local mode uses FastAPI; deployed version loads model directly — see live demo link above)

## Key Design Decisions
- **Recall prioritised over accuracy** — class imbalance (78/22 split); false negatives costlier than false positives
- **SMOTE on training data only** — prevents data leakage
- **ETL over ELT** — transformations applied before loading to warehouse
- **Two databases** — ColumnStore for analytical queries; PostgreSQL for row-level prediction logging
- **Live drift monitoring** — Evidently recall fed into Redis; Airflow retrain triggers on degradation
