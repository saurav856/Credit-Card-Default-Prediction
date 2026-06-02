import pickle
import redis
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score
import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
mlflow.set_tracking_uri("file:///opt/airflow/mlruns")

print("Loading preprocessed data from Redis...")
r = redis.Redis(host="mlops_redis", port=6379)
X_train = pickle.loads(r.get("X_train"))
X_test = pickle.loads(r.get("X_test"))
y_train = pickle.loads(r.get("y_train"))
y_test = pickle.loads(r.get("y_test"))
scaler = pickle.loads(r.get("scaler"))

mlflow.set_experiment("credit_card_default")

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
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        mlflow.log_param("model", name)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auc_roc", auc)
        mlflow.sklearn.log_model(model, name)
        print(f"{name} — Recall: {recall:.4f} | AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        if recall > best_recall:
            best_recall = recall
            best_model = model
            best_model_name = name
            best_run_id = run.info.run_id

print(f"\nBest model: {best_model_name} with Recall: {best_recall:.4f}")

# Register best model in MLflow Model Registry
print("Registering best model in MLflow Model Registry...")
model_uri = f"runs:/{best_run_id}/{best_model_name}"
model_info = mlflow.register_model(
    model_uri=model_uri,
    name="CreditCardDefaultModel"
)
print(f"Model registered: {model_info.name} version {model_info.version}")

# Cache to Redis for monitor.py
r.set("best_model", pickle.dumps(best_model))
r.set("scaler", pickle.dumps(scaler))
r.set("registered_model_name", b"CreditCardDefaultModel")
r.set("registered_model_version", str(model_info.version).encode())
print("Best model and scaler cached to Redis.")

# Save to disk as backup
os.makedirs("/opt/airflow/models", exist_ok=True)
pickle.dump(best_model, open("/opt/airflow/models/best_model.pkl", "wb"))
print("Best model saved to disk.")
