# ======================================================================
# XGBOOST + FUZZY + SAVE PNG IMAGES + PDF REPORT (in xgboost/ folder)
# ======================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import skfuzzy as fuzz
import pickle

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ======================================================================
# 0. CREATE OUTPUT FOLDER "xgboost"
# ======================================================================

os.makedirs("xgboost", exist_ok=True)


# ======================================================================
# 1. LOAD DATASET (KAGGLE)
# ======================================================================

df = pd.read_csv("heart.csv")


# ======================================================================
# 2. BINARY TARGET
# ======================================================================

df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


# ======================================================================
# 3. FEATURES
# ======================================================================

features = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
]

X = df[features]
y = df['target']


# ======================================================================
# 4. TRAIN–TEST SPLIT
# ======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# ======================================================================
# 5. SCALING
# ======================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ======================================================================
# 6. XGBOOST MODEL (95–98% expected accuracy)
# ======================================================================

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


# ======================================================================
# 7. SAVE XGBOOST MODEL
# ======================================================================

with open("xgboost/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)


# ======================================================================
# 8. CONFUSION MATRIX + SAVE PNG
# ======================================================================

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("xgboost/confusion_matrix.png", dpi=300)
plt.close()


# ======================================================================
# 9. FUZZY C-MEANS CLUSTERING
# ======================================================================

fuzzy_features = ["cp", "thalach", "oldpeak", "exang", "ca", "slope", "thal"]
X_fuzzy = df[fuzzy_features]
data = X_fuzzy.values.T

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data, c=2, m=2.0, error=0.005, maxiter=1000
)

cluster_labels = np.argmax(u, axis=0)

df_compare = pd.DataFrame({
    "cluster": cluster_labels,
    "target": y.values
})

mapping = {}
for c in [0, 1]:
    mapping[c] = df_compare[df_compare["cluster"] == c]["target"].mode()[0]

mapped = df_compare["cluster"].map(mapping)
alignment = (mapped.values == y.values).mean()


# ======================================================================
# 10. CORRECT FUZZY PNGs (3 proper fuzzy plots)
# ======================================================================

# -------- Fuzzy Plot 1: membership strength --------
plt.figure(figsize=(10,5))
plt.plot(u[0], label="Membership to Cluster 0 (u0)")
plt.plot(u[1], label="Membership to Cluster 1 (u1)")
plt.title("Fuzzy Membership Strength (u0 & u1)")
plt.xlabel("Sample Index")
plt.ylabel("Membership Value")
plt.legend()
plt.savefig("xgboost/fuzzy_membership_strength.png", dpi=300)
plt.close()

# -------- Fuzzy Plot 2: feature space clustering --------
plt.figure(figsize=(7,5))
plt.scatter(X_fuzzy["cp"], X_fuzzy["oldpeak"], c=cluster_labels, cmap="coolwarm", alpha=0.7)
plt.xlabel("Chest Pain Type (cp)")
plt.ylabel("Oldpeak")
plt.title("Fuzzy Clusters in Feature Space (cp vs oldpeak)")
plt.savefig("xgboost/fuzzy_feature_space.png", dpi=300)
plt.close()

# -------- Fuzzy Plot 3: intensity heatmap (risk) --------
plt.figure(figsize=(7,5))
plt.scatter(X_fuzzy["thalach"], X_fuzzy["oldpeak"], c=u[1], cmap="viridis", alpha=0.8)
plt.colorbar(label="Membership in Cluster 1 (Disease Risk)")
plt.xlabel("Thalach (Heart Rate)")
plt.ylabel("Oldpeak")
plt.title("Fuzzy Membership Intensity (Risk Heatmap)")
plt.savefig("xgboost/fuzzy_membership_intensity.png", dpi=300)
plt.close()


# ======================================================================
# 11. SYSTEM ARCHITECTURE PNG
# ======================================================================

plt.figure(figsize=(8,5))
plt.text(0.1, 0.85, "Input Data (Kaggle Heart Dataset)", fontsize=12)
plt.text(0.1, 0.7, "↓ Preprocessing (Scaling)", fontsize=12)
plt.text(0.1, 0.55, "↓ XGBoost Classifier", fontsize=12)
plt.text(0.1, 0.4, "↓ Confusion Matrix", fontsize=12)
plt.text(0.1, 0.25, "↓ Fuzzy C-Means (Validation)", fontsize=12)
plt.axis("off")
plt.savefig("xgboost/system_architecture.png", dpi=300)
plt.close()


# ======================================================================
# 12. GENERATE PDF REPORT
# ======================================================================

c = canvas.Canvas("xgboost/full_report.pdf", pagesize=letter)
c.setFont("Helvetica", 12)

c.drawString(30, 750, "Heart Disease Prediction Report - XGBoost")
c.drawString(30, 730, f"XGBoost Accuracy: {accuracy_score(y_test, pred):.4f}")
c.drawString(30, 710, f"Fuzzy Alignment: {alignment:.4f}")
c.drawString(30, 690, f"Fuzzy Partition Coefficient (FPC): {fpc:.4f}")

c.drawString(30, 660, "Saved Files:")
c.drawString(40, 640, "xgb_model.pkl")
c.drawString(40, 620, "confusion_matrix.png")
c.drawString(40, 600, "fuzzy_membership_strength.png")
c.drawString(40, 580, "fuzzy_feature_space.png")
c.drawString(40, 560, "fuzzy_membership_intensity.png")
c.drawString(40, 540, "system_architecture.png")
c.drawString(40, 520, "full_report.pdf")

c.save()


# ======================================================================
# DONE
# ======================================================================

print("\nAll XGBoost outputs saved inside the 'xgboost/' folder!")
print("Accuracy:", accuracy_score(y_test, pred))
print("Fuzzy Alignment:", alignment)
