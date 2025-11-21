# ============================================================
# INTERPRETABLE AI: SHAP ANALYSIS ON SYNTHETIC CREDIT RISK MODEL
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
import shap
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# 1. SYNTHETIC HIGH-DIMENSIONAL FINANCIAL DATA (40 FEATURES)
# -------------------------------------------------------------
np.random.seed(42)
n_samples = 5000

# Realistic numeric financial features
df = pd.DataFrame({
    "credit_score": np.random.normal(680, 60, n_samples).clip(300, 850),
    "annual_income": np.random.lognormal(10, 0.5, n_samples),
    "loan_amount": np.random.uniform(2000, 35000, n_samples),
    "interest_rate": np.random.uniform(5, 30, n_samples),
    "dti_ratio": np.random.uniform(0, 40, n_samples),
    "num_open_accounts": np.random.randint(1, 20, n_samples),
    "revolving_utilization": np.random.uniform(1, 150, n_samples),
    "num_delinquencies": np.random.poisson(0.3, n_samples),
    "employment_years": np.random.randint(0, 30, n_samples),
    "inq_last_6m": np.random.poisson(1, n_samples)
})

# Categorical fields
df["home_ownership"] = np.random.choice(["OWN", "MORTGAGE", "RENT"], n_samples)
df["loan_purpose"] = np.random.choice(["credit_card", "debt_consolidation", "home", "car", "other"], n_samples)
df["state"] = np.random.choice(["CA", "TX", "NY", "FL", "IL", "PA"], n_samples)

# Add 27 synthetic features
for i in range(27):
    df[f"synthetic_feature_{i+1}"] = np.random.normal(0, 1, n_samples)

# Target variable – risk increases with financial stress
risk_score = (
    (df["interest_rate"] * 0.04) +
    (df["dti_ratio"] * 0.03) +
    (df["num_delinquencies"] * 0.20) -
    (df["credit_score"] * 0.002) +
    np.random.normal(0, 0.5, n_samples)
)

df["default"] = (risk_score > risk_score.mean()).astype(int)

# -------------------------------------------------------------
# 2. PREPROCESSING PIPELINE
# -------------------------------------------------------------
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
numeric_features.remove("default")

categorical_features = ["home_ownership", "loan_purpose", "state"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# PCA for dimensionality reduction
pca = PCA(n_components=20)

# -------------------------------------------------------------
# 3. TRAIN–TEST SPLIT
# -------------------------------------------------------------
X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -------------------------------------------------------------
# 4. XGBOOST MODEL
# -------------------------------------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss"
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("pca", pca),
    ("model", xgb_model)
])

pipeline.fit(X_train, y_train)

# -------------------------------------------------------------
# 5. MODEL EVALUATION
# -------------------------------------------------------------
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("MODEL PERFORMANCE:")
print("AUC:", auc)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

# -------------------------------------------------------------
# 6. SHAP GLOBAL EXPLANATION
# -------------------------------------------------------------
# SHAP must use transformed data
X_transformed = pipeline.named_steps["preprocess"].transform(X_test)
X_pca = pipeline.named_steps["pca"].transform(X_transformed)

explainer = shap.Explainer(pipeline.named_steps["model"])
shap_values = explainer(X_pca)

# Generate global plots
shap.summary_plot(shap_values, X_pca)
shap.summary_plot(shap_values, X_pca, plot_type="bar")

# -------------------------------------------------------------
# 7. SELECT THREE BORROWERS (Corrected Method)
# -------------------------------------------------------------
test_df = X_test.copy()
test_df["prob"] = y_pred_prob

# Reset index to match SHAP order
test_df = test_df.reset_index(drop=True)

# Identify positions (SHAP uses same row positions)
low_pos = test_df["prob"].idxmin()
high_pos = test_df["prob"].idxmax()
border_pos = (test_df["prob"] - 0.50).abs().idxmin()

# -------------------------------------------------------------
# 8. SHAP LOCAL FORCE PLOTS (Correct + Error-Free)
# -------------------------------------------------------------
shap.force_plot(
    explainer.expected_value,
    shap_values.values[high_pos],
    matplotlib=True
)

shap.force_plot(
    explainer.expected_value,
    shap_values.values[low_pos],
    matplotlib=True
)

shap.force_plot(
    explainer.expected_value,
    shap_values.values[border_pos],
    matplotlib=True
)

