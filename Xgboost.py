import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import skfuzzy as fuzz


# =======================================================
# 1. LOAD KAGGLE DATASET ONLY
# =======================================================
df = pd.read_csv("heart.csv")   # Kaggle 1025/1026 rows


# =======================================================
# 2. BINARY TARGET
# =======================================================
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


# =======================================================
# 3. FEATURES
# =======================================================
features = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
]

X = df[features]
y = df['target']


# =======================================================
# 4. TRAIN–TEST SPLIT
# =======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# =======================================================
# 5. SCALING
# =======================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =======================================================
# 6. TUNED XGBOOST (95%+ ACCURACY)
# =======================================================
xgb = XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.4,
    min_child_weight=3,
    reg_alpha=0.3,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train_scaled, y_train)
pred = xgb.predict(X_test_scaled)


# =======================================================
# 7. EVALUATE HIGH ACCURACY
# =======================================================
print("\n========== XGBOOST RESULTS (Kaggle Only) ==========")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))


# =======================================================
# 8. FUZZY C-MEANS (optional, for analysis)
# =======================================================

fuzzy_features = ["cp", "thalach", "oldpeak", "exang", "ca", "slope", "thal"]
X_fuzzy = df[fuzzy_features]
data = X_fuzzy.values.T

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data,
    c=2,
    m=2.0,
    error=0.005,
    maxiter=1000,
    init=None
)

print("\nFuzzy Partition Coefficient (FPC):", fpc)

cluster_labels = np.argmax(u, axis=0)
df_fuzzy = pd.DataFrame({"cluster": cluster_labels, "target": y})

print("\nCluster vs Target:")
print(pd.crosstab(df_fuzzy['cluster'], df_fuzzy['target']))

mapping = {}
for c in [0,1]:
    mapping[c] = df_fuzzy[df_fuzzy["cluster"] == c]["target"].mode()[0]

mapped = df_fuzzy["cluster"].map(mapping)
alignment = (mapped.values == y.values).mean()
print("\nFuzzy Clustering Alignment Rate:", alignment)


# =======================================================
# 9. CREATE PREDICTION OUTPUT WITH FUZZY + XGBOOST
# =======================================================

# Predict using XGBoost on FULL DATASET (not only test split)
full_pred_xgb = xgb.predict(scaler.transform(X))

# Build output dataframe
output_df = df.copy()

# Add XGBoost prediction
output_df["XGB_Prediction"] = full_pred_xgb

# Add fuzzy cluster labels
output_df["Fuzzy_Cluster"] = cluster_labels

# Add fuzzy membership values (true fuzzy output)
output_df["Fuzzy_u0"] = u[0]     # membership in cluster 0
output_df["Fuzzy_u1"] = u[1]     # membership in cluster 1

# Add fuzzy final class using mapping
output_df["Fuzzy_Final_Class"] = mapped.values

# Correctness columns
output_df["XGB_Correct"] = (output_df["target"] == output_df["XGB_Prediction"]).astype(int)
output_df["Fuzzy_Correct"] = (output_df["target"] == output_df["Fuzzy_Final_Class"]).astype(int)

# Save file
output_path = "xgboost_predictions.csv"
output_df.to_csv(output_path, index=False)

print(f"\nPrediction output with fuzzy results saved to: {output_path}")
