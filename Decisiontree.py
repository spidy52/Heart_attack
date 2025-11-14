# -------------------------------------------------------
# DECISION TREE (Kaggle Only) + Fuzzy C-Means + Prediction CSV
# -------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import skfuzzy as fuzz


# =======================================================
# 1. LOAD DATASET (KAGGLE)
# =======================================================

df = pd.read_csv("heart.csv")   # Kaggle dataset


# =======================================================
# 2. BINARY TARGET
# =======================================================

df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


# =======================================================
# 3. SELECT FEATURES
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
# 5. SCALING (not needed for DT but for fuzzy consistency)
# =======================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =======================================================
# 6. DECISION TREE MODEL
# =======================================================

dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)


# =======================================================
# 7. EVALUATE DECISION TREE
# =======================================================

print("\n========== DECISION TREE RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, dt_pred))


# =======================================================
# 8. PRINT DECISION TREE RULES
# =======================================================

tree_rules = export_text(dt, feature_names=features)
print("\n========== DECISION TREE RULES ==========")
print(tree_rules)


# =======================================================
# 9. FUZZY C-MEANS (KAGGLE ONLY)
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


# =======================================================
# 10. ALIGN FUZZY CLUSTERS TO TRUE LABELS
# =======================================================

cluster_labels = np.argmax(u, axis=0)

df_compare = pd.DataFrame({
    "cluster": cluster_labels,
    "target": y.values
})

print("\nCluster vs Target:")
print(pd.crosstab(df_compare["cluster"], df_compare["target"]))

# Map clusters to correct classes
mapping = {}
for c in [0, 1]:
    mapping[c] = df_compare[df_compare["cluster"] == c]["target"].mode()[0]

mapped = df_compare["cluster"].map(mapping)
alignment = (mapped.values == y.values).mean()

print("\nFuzzy Clustering Alignment Rate:", alignment)


# =======================================================
# 11. CREATE PREDICTION OUTPUT CSV (DT + Fuzzy)
# =======================================================

# Decision Tree predictions on FULL dataset
full_pred_dt = dt.predict(X)

# Build output dataframe
output_df = df.copy()

output_df["DT_Prediction"] = full_pred_dt

# Fuzzy outputs
output_df["Fuzzy_Cluster"] = cluster_labels
output_df["Fuzzy_u0"] = u[0]
output_df["Fuzzy_u1"] = u[1]
output_df["Fuzzy_Final_Class"] = mapped.values

# Correctness
output_df["DT_Correct"] = (output_df["target"] == output_df["DT_Prediction"]).astype(int)
output_df["Fuzzy_Correct"] = (output_df["target"] == output_df["Fuzzy_Final_Class"]).astype(int)

# Save CSV
output_path = "decision_tree_predictions.csv"
output_df.to_csv(output_path, index=False)

print(f"\nPrediction output saved to: {output_path}")
