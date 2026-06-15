import great_expectations as gx
import pandas as pd
import pickle
import redis
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# --- Load from Redis ---
print("Loading data from Redis...")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
r = redis.Redis(host=REDIS_HOST, port=6379)
df = pickle.loads(r.get("raw_data"))
print(f"Loaded from Redis: {df.shape}")

# --- Great Expectations Validation (GE 1.4.4 Fluent API) ---
print("Running Great Expectations validation...")
context = gx.get_context(mode="ephemeral")
datasource = context.data_sources.add_pandas(name="credit_pandas_datasource")
data_asset = datasource.add_dataframe_asset(name="credit_card_data")
batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

suite = context.suites.add(gx.ExpectationSuite(name="credit_card_suite"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="default_payment"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="default_payment", value_set=[0, 1]))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="SEX", value_set=[1, 2]))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="EDUCATION", value_set=[0, 1, 2, 3, 4, 5, 6]))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="MARRIAGE", value_set=[0, 1, 2, 3]))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="AGE", min_value=18, max_value=100))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="LIMIT_BAL", min_value=0))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="BILL_AMT1"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="PAY_AMT1", min_value=0))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="PAY_0", min_value=-2, max_value=8))

validation_definition = context.validation_definitions.add(
    gx.ValidationDefinition(name="credit_validation", data=batch_definition, suite=suite)
)
result = validation_definition.run(batch_parameters={"dataframe": df})
print("Validation success:", result.success)
for res in result.results:
    print(f"  {res.expectation_config.type} | Success: {res.success}")

# --- Preprocessing ---
print("Preprocessing data...")
FEATURES = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3",
            "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]
TARGET = "default_payment"

X = df[FEATURES]
y = df[TARGET]

print("Splitting data before SMOTE...")
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size before SMOTE: {X_train_raw.shape} | Test size: {X_test_raw.shape}")

print("Applying SMOTE to training data only...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_raw, y_train_raw)
print(f"Train after SMOTE: {X_train_res.shape}")

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test_raw)

print("Caching preprocessed data to Redis...")
r.set("X_train", pickle.dumps(X_train_scaled))
r.set("X_test", pickle.dumps(X_test_scaled))
r.set("y_train", pickle.dumps(y_train_res))
r.set("y_test", pickle.dumps(y_test_raw))
r.set("scaler", pickle.dumps(scaler))
print("Preprocessing complete.")
print(f"Final train: {X_train_scaled.shape} | Final test: {X_test_scaled.shape}")