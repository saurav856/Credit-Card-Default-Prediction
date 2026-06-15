import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy
from scipy import stats

warnings.filterwarnings("ignore")

# Config
XLS_PATH   = os.path.expanduser("~/mlops/data/default of credit card clients.xls")
DB_URL     = "mysql+pymysql://mlops_user:mlops_pass@127.0.0.1:3306/creditcard_db"
OUT_DIR    = os.path.expanduser("~/mlops/eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES   = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
              "PAY_0", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"]
TARGET     = "default_payment_next_month"

# 1. Load raw source file 
print("\n=== Loading raw XLS ===")
raw = pd.read_excel(XLS_PATH, header=1)
raw.columns = raw.columns.str.strip().str.lower().str.replace(".", "_", regex=False).str.replace(" ", "_")
raw.rename(columns={"default_payment_next_month": TARGET}, inplace=True)

# Keep only our 9 features + target + ID
keep = ["id"] + [f.lower() for f in FEATURES] + [TARGET]
# normalise column names to lowercase for safety
raw.columns = [c.lower() for c in raw.columns]
df_source = raw[keep].copy()
df_source.columns = [c.upper() if c != TARGET else TARGET for c in df_source.columns]
df_source.drop(columns=["ID"], inplace=True)

print(f"Source rows: {len(df_source):,} | cols: {df_source.shape[1]}")

# 2. Load stored MariaDB data 
print("\n=== Loading from MariaDB ===")
try:
    engine = sqlalchemy.create_engine(DB_URL)
    with engine.connect() as conn:
        df_db = pd.read_sql("SELECT * FROM fact_credit_default", conn)
    print(f"DB rows: {len(df_db):,} | cols: {df_db.shape[1]}")
    db_available = True
except Exception as e:
    print(f"MariaDB not reachable: {e}")
    print("Using source data as proxy for DB data.")
    df_db = df_source.copy()
    db_available = False

# Work on source for EDA
df = df_source.copy()

# 3. Variable types & measurement levels
print("\n=== Variable Types & Measurement Levels ===")

var_meta = {
    "LIMIT_BAL": ("Continuous", "Ratio",    "Credit limit in NT dollars — true zero exists"),
    "SEX":       ("Nominal",    "Nominal",   "1=Male, 2=Female — no ordering"),
    "EDUCATION": ("Ordinal",    "Ordinal",   "1=grad, 2=uni, 3=high school, 4=other — ordered"),
    "MARRIAGE":  ("Nominal",    "Nominal",   "1=married, 2=single, 3=other — no ordering"),
    "AGE":       ("Discrete",   "Ratio",     "Age in years — true zero, integer counts"),
    "PAY_0":     ("Ordinal",    "Ordinal",   "Repayment status -2 to 8 — ordered scale"),
    "BILL_AMT1": ("Continuous", "Interval",  "Bill amount — can be negative (credit balance)"),
    "PAY_AMT1":  ("Continuous", "Ratio",     "Payment amount — true zero (no payment)"),
    "PAY_AMT2":  ("Continuous", "Ratio",     "Payment amount — true zero (no payment)"),
}

type_rows = []
for var, (dtype, level, note) in var_meta.items():
    pandas_dtype = df[var].dtype
    type_rows.append([var, dtype, level, str(pandas_dtype), note])

type_df = pd.DataFrame(type_rows,
    columns=["Variable", "Statistical Type", "Measurement Level", "Pandas dtype", "Notes"])
print(type_df.to_string(index=False))
type_df.to_csv(os.path.join(OUT_DIR, "variable_types.csv"), index=False)

# 4. Missing values
print("\n=== Missing Values ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print(missing_df[missing_df["missing_count"] > 0])
if missing_df["missing_count"].sum() == 0:
    print("No missing values found in 9 selected features + target.")
missing_df.to_csv(os.path.join(OUT_DIR, "missing_values.csv"))

# 5. Outlier detection
print("\n=== Outlier Detection (IQR + Z-score) ===")
continuous = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"]
outlier_rows = []

for col in continuous:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_iqr = ((df[col] < lb) | (df[col] > ub)).sum()
    z = np.abs(stats.zscore(df[col].dropna()))
    n_z = (z > 3).sum()
    outlier_rows.append({
        "Variable": col,
        "IQR_outliers": int(n_iqr),
        "IQR_pct": round(n_iqr / len(df) * 100, 2),
        "Zscore_outliers_gt3": int(n_z),
        "Min": df[col].min(),
        "Max": df[col].max(),
        "Mean": round(df[col].mean(), 2),
        "Median": df[col].median(),
    })
    print(f"{col}: IQR outliers={n_iqr} ({round(n_iqr/len(df)*100,2)}%), Z>3={n_z}")

outlier_df = pd.DataFrame(outlier_rows)
outlier_df.to_csv(os.path.join(OUT_DIR, "outliers.csv"), index=False)

# Categorical value checks
print("\n--- Categorical value range checks ---")
cat_checks = {
    "SEX":       (1, 2),
    "EDUCATION": (0, 6),
    "MARRIAGE":  (0, 3),
    "PAY_0":     (-2, 8),
}
for col, (lo, hi) in cat_checks.items():
    out_of_range = df[(df[col] < lo) | (df[col] > hi)]
    unique_vals = sorted(df[col].unique())
    print(f"{col}: unique={unique_vals} | out-of-range={len(out_of_range)}")

# EDUCATION=0 and MARRIAGE=0 check (undocumented categories)
edu_zero = (df["EDUCATION"] == 0).sum()
mar_zero = (df["MARRIAGE"] == 0).sum()
print(f"EDUCATION=0 (undocumented): {edu_zero}")
print(f"MARRIAGE=0  (undocumented): {mar_zero}")

# 6. Descriptive statistics
print("\n=== Descriptive Statistics ===")
desc = df.describe().round(2)
print(desc)
desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"))

# 7. Distributions — plots
print("\n=== Plotting Distributions ===")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(FEATURES):
    ax = axes[i]
    if col in ["SEX", "EDUCATION", "MARRIAGE", "PAY_0"]:
        vc = df[col].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color="steelblue", edgecolor="white")
        ax.set_title(f"{col} (ordinal/nominal)")
    else:
        ax.hist(df[col], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(df[col].mean(),   color="red",    linestyle="--", label="mean")
        ax.axvline(df[col].median(), color="orange", linestyle="--", label="median")
        ax.legend(fontsize=7)
        ax.set_title(f"{col} (continuous)")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

plt.suptitle("Feature Distributions — UCI Credit Card Default Dataset", fontsize=13, y=1.01)
plt.tight_layout()
dist_path = os.path.join(OUT_DIR, "distributions.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {dist_path}")

# 8. Class balance
print("\n=== Target Class Balance ===")
class_counts = df[TARGET].value_counts()          
class_pct    = df[TARGET].value_counts(normalize=True).round(4) * 100
print(class_counts)
print(class_pct)

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["No Default (0)", "Default (1)"],
       [class_counts[0], class_counts[1]],
       color=["steelblue", "tomato"], edgecolor="white")
ax.set_title("Target Variable Distribution", pad=14)
ax.set_ylabel("Count")
for i, v in enumerate([class_counts[0], class_counts[1]]):
    ax.text(i, v + 500,
            f"{v:,}\n({class_pct.iloc[i]:.1f}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, max(class_counts[0], class_counts[1]) * 1.18)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_balance.png"), dpi=150)
plt.close()

# 9. Feature–feature correlations 
print("\n=== Feature–Feature Correlations (Pearson) ===")
corr_matrix = df[FEATURES].corr().round(3)
print(corr_matrix)
corr_matrix.to_csv(os.path.join(OUT_DIR, "feature_correlations.csv"))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Feature–Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_correlation_heatmap.png"), dpi=150)
plt.close()
print(f"Saved correlation heatmap.")

# 10. Feature–target correlations
print("\n=== Feature–Target Correlations ===")
point_biserial = {}
for col in FEATURES:
    corr, pval = stats.pointbiserialr(df[col], df[TARGET])
    point_biserial[col] = {"correlation": round(corr, 4), "p_value": round(pval, 6)}
    print(f"{col}: r={corr:.4f}, p={pval:.4f}")

pb_df = pd.DataFrame(point_biserial).T.sort_values("correlation", ascending=False)
pb_df.to_csv(os.path.join(OUT_DIR, "feature_target_correlations.csv"))

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["tomato" if v > 0 else "steelblue" for v in pb_df["correlation"]]
ax.barh(pb_df.index, pb_df["correlation"], color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Point-Biserial Correlation: Each Feature vs Default Target")
ax.set_xlabel("Correlation coefficient")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_target_correlations.png"), dpi=150)
plt.close()

# 11. Default rate by categorical variable
print("\n=== Default Rate by Categorical Variable ===")
cat_vars = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(cat_vars):
    rate = df.groupby(col)[TARGET].mean().round(4) * 100
    count = df.groupby(col)[TARGET].count()
    print(f"\n{col} default rate (%):\n{rate}")
    axes[i].bar(rate.index.astype(str), rate.values, color="tomato", edgecolor="white")
    axes[i].set_title(f"Default Rate by {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Default rate (%)")

plt.suptitle("Default Rate by Categorical Features", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "default_rate_by_category.png"), dpi=150)
plt.close()

# 12. Box plots: continuous features by target
print("\n=== Box Plots: Continuous Features by Target ===")
cont_vars = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"]

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, col in enumerate(cont_vars):
    df.boxplot(column=col, by=TARGET, ax=axes[i], grid=False)
    axes[i].set_title(col)
    axes[i].set_xlabel("Default (0=No, 1=Yes)")

plt.suptitle("Continuous Features by Default Status")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplots_by_target.png"), dpi=150)
plt.close()

# 13. Source vs MariaDB comparison
print("\n=== Source vs Stored Data Comparison ===")

comparison = {
    "source_rows": len(df_source),
    "db_rows":     len(df_db),
    "row_diff":    len(df_source) - len(df_db),
}

if db_available:
    try:
        # LIMIT_BAL lives in fact_credit_default
        with engine.connect() as conn:
            db_fact = pd.read_sql("SELECT LIMIT_BAL FROM fact_credit_default", conn)
        src_mean_lb = df_source["LIMIT_BAL"].mean()
        db_mean_lb  = db_fact["LIMIT_BAL"].mean()
        comparison["LIMIT_BAL_source_mean"] = round(src_mean_lb, 2)
        comparison["LIMIT_BAL_db_mean"]     = round(db_mean_lb, 2)
        comparison["LIMIT_BAL_diff"]        = round(src_mean_lb - db_mean_lb, 4)
    except Exception as e:
        comparison["LIMIT_BAL_error"] = str(e)

    try:
        # AGE lives in dim_client
        with engine.connect() as conn:
            db_dim = pd.read_sql("SELECT AGE FROM dim_client", conn)
        src_mean_age = df_source["AGE"].mean()
        db_mean_age  = db_dim["AGE"].mean()
        comparison["AGE_source_mean"] = round(src_mean_age, 2)
        comparison["AGE_db_mean"]     = round(db_mean_age, 2)
        comparison["AGE_diff"]        = round(src_mean_age - db_mean_age, 4)
    except Exception as e:
        comparison["AGE_error"] = str(e)

    try:
        # Row count from full JOIN
        with engine.connect() as conn:
            db_full = pd.read_sql("""
                SELECT f.LIMIT_BAL, f.BILL_AMT1, f.default_payment,
                       c.SEX, c.EDUCATION, c.MARRIAGE, c.AGE,
                       r.PAY_0, r.PAY_AMT1, r.PAY_AMT2
                FROM fact_credit_default f
                JOIN dim_client c ON f.client_id = c.client_id
                JOIN dim_repayment r ON f.repayment_id = r.repayment_id
            """, conn)
        comparison["db_rows"]  = len(db_full)
        comparison["row_diff"] = len(df_source) - len(db_full)
        print(f"Full JOIN row count: {len(db_full):,}")
    except Exception as e:
        comparison["join_error"] = str(e)
else:
    comparison["note"] = "MariaDB not reachable — comparison skipped"

print(comparison)
pd.DataFrame([comparison]).to_csv(os.path.join(OUT_DIR, "source_vs_db.csv"), index=False)

# 14. Skewness & Kurtosis
print("\n=== Skewness & Kurtosis ===")
sk_data = []
for col in continuous:
    skew = round(df[col].skew(), 4)
    kurt = round(df[col].kurtosis(), 4)
    sk_data.append({"Variable": col, "Skewness": skew, "Kurtosis": kurt})
    print(f"{col}: skew={skew}, kurtosis={kurt}")

sk_df = pd.DataFrame(sk_data)
sk_df.to_csv(os.path.join(OUT_DIR, "skewness_kurtosis.csv"), index=False)

# Summary 
print(f"""
=== EDA Complete ===
All outputs saved to: {OUT_DIR}
Files:
  variable_types.csv
  missing_values.csv
  outliers.csv
  descriptive_stats.csv
  skewness_kurtosis.csv
  feature_correlations.csv
  feature_target_correlations.csv
  source_vs_db.csv
  distributions.png
  class_balance.png
  feature_correlation_heatmap.png
  feature_target_correlations.png
  default_rate_by_category.png
  boxplots_by_target.png
""")