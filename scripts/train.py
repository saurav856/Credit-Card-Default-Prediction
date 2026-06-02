import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score
import os

print("Loading preprocessed data...")
X_train = pickle.load(open("models/X_train.pkl", "rb"))
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_train = pickle.load(open("models/y_train.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))

mlflow.set_experiment("credit_card_default")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_recall = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
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

print(f"\nBest model: {best_model_name} with Recall: {best_recall:.4f}")
pickle.dump(best_model, open("models/best_model.pkl", "wb"))
print("Best model saved to models/best_model.pkl")