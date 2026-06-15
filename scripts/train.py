import pickle
import redis
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score
import pandas as pd
import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
r = redis.Redis(host=REDIS_HOST, port=6379)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///home/blond/mlops/mlruns")
mlflow.set_tracking_uri(MLFLOW_URI)

print("Loading preprocessed data from Redis...")
X_train = pickle.loads(r.get("X_train"))
X_test  = pickle.loads(r.get("X_test"))
y_train = pickle.loads(r.get("y_train"))
y_test  = pickle.loads(r.get("y_test"))
scaler  = pickle.loads(r.get("scaler"))

mlflow.set_experiment("credit_card_default")

FEATURE_NAMES = [
    'LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3',
    'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2'
]

THRESHOLD = 0.40 

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_recall = 0
best_model = None
best_model_name = ""
best_run_id = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        y_pred = (y_prob >= THRESHOLD).astype(int)

        recall = recall_score(y_test, y_pred, pos_label=1)
        auc    = roc_auc_score(y_test, y_prob)

        # log params
        mlflow.log_param("model", name)
        mlflow.log_param("threshold", THRESHOLD)

        # log metrics
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auc_roc", auc)

        # log model artifact
        mlflow.sklearn.log_model(model, name)

        report_str = classification_report(y_test, y_pred)
        print(f"{name} — Recall: {recall:.4f} | AUC: {auc:.4f}")
        print(report_str)
        mlflow.log_text(report_str, f"{name}_classification_report.txt")

        if name == "LogisticRegression":
            coef_df = pd.DataFrame({
                'feature': FEATURE_NAMES,
                'coefficient': model.coef_[0]
            }).sort_values('coefficient', ascending=False)
            print("\nLogistic Regression Coefficients:")
            print(coef_df.to_string(index=False))
            mlflow.log_text(coef_df.to_string(index=False), "lr_coefficients.txt")

        if name == "RandomForest":
            fi_df = pd.DataFrame({
                'feature': FEATURE_NAMES,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nRandom Forest Feature Importances:")
            print(fi_df.to_string(index=False))
            mlflow.log_text(fi_df.to_string(index=False), "rf_feature_importance.txt")

        if recall > best_recall:
            best_recall     = recall
            best_model      = model
            best_model_name = name
            best_run_id     = run.info.run_id

print(f"\nBest model: {best_model_name} with Recall: {best_recall:.4f}")
print("Registering best model in MLflow Model Registry...")

model_uri  = f"runs:/{best_run_id}/{best_model_name}"
model_info = mlflow.register_model(
    model_uri=model_uri,
    name="CreditCardDefaultModel"
)
print(f"Model registered: {model_info.name} version {model_info.version}")

r.set("best_model",               pickle.dumps(best_model))
r.set("scaler",                   pickle.dumps(scaler))
r.set("registered_model_name",    b"CreditCardDefaultModel")
r.set("registered_model_version", str(model_info.version).encode())
print("Best model and scaler cached to Redis.")

MODEL_DIR = os.getenv("MODEL_DIR", "/home/blond/mlops/models")
os.makedirs(MODEL_DIR, exist_ok=True)
print("Best model saved to disk.")

r.set("latest_recall", str(best_recall))
print(f"Latest recall pushed to Redis: {best_recall:.4f}")