import pandas as pd
import pickle
import redis
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import great_expectations as gx

# --- Load from Redis ---
print("Loading data from Redis...")
r = redis.Redis(host="127.0.0.1", port=6379)
df = pickle.loads(r.get("raw_data"))
print(f"Loaded from Redis: {df.shape}")

# --- Great Expectations Validation ---
print("Running Great Expectations validation...")
context = gx.get_context(mode="ephemeral")
ds = context.sources.add_pandas("pandas_source")
da = ds.add_dataframe_asset("credit_data")
batch = da.build_batch_request(dataframe=df)
suite = context.add_expectation_suite("credit_suite")

validator = context.get_validator(
    batch_request=batch,
    expectation_suite=suite
)

validator.expect_column_values_to_not_be_null("default_payment")
validator.expect_column_values_to_be_between("PAY_0", min_value=-2, max_value=8)
validator.expect_column_values_to_be_between("LIMIT_BAL", min_value=0)
validator.expect_column_values_to_be_in_set("default_payment", [0, 1])

results = validator.validate()
print(f"Validation passed: {results['success']}")

# --- Preprocessing ---
df.drop(columns=["ID"], inplace=True)
X = df.drop(columns=["default_payment"])
y = df["default_payment"]

print(f"Class distribution BEFORE SMOTE:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(f"Class distribution AFTER SMOTE:\n{pd.Series(y_train_res).value_counts()}")

# --- Save to disk ---
os.makedirs("models", exist_ok=True)
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(X_train_res, open("models/X_train.pkl", "wb"))
pickle.dump(X_test_scaled, open("models/X_test.pkl", "wb"))
pickle.dump(y_train_res, open("models/y_train.pkl", "wb"))
pickle.dump(y_test, open("models/y_test.pkl", "wb"))

# --- Cache processed data to Redis ---
r.set("X_train", pickle.dumps(X_train_res))
r.set("y_train", pickle.dumps(y_train_res))
r.set("X_test", pickle.dumps(X_test_scaled))
r.set("y_test", pickle.dumps(y_test))

print("Preprocessing complete. Data cached to Redis.")