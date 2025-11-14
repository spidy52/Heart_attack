# ======================================================================
# DECISION TREE + CORRECTED FUZZY OUTPUT + PNG + PDF (output/ folder)
# ======================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import skfuzzy as fuzz
import pickle

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ======================================================================
# 0. CREATE OUTPUT FOLDER
# ======================================================================

os.makedirs("output", exist_ok=True)


# ======================================================================
# 1. LOAD DATASET
# ======================================================================

df = pd.read_csv("heart.csv")   # Kaggle dataset


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
# 6. DECISION TREE TRAINING
# ======================================================================

dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)


# ======================================================================
# 7. SAVE DECISION TREE MODEL
# ======================================================================

with open("output/decision_tree.pkl", "wb") as f:
    pickle.dump(dt, f)


# ======================================================================
# 8. SAVE DECISION TREE RULES
# ======================================================================

rules = export_text(dt, feature_names=features)
with open("output/tree_rules.txt", "w") as f:
    f.write(rules)


# ======================================================================
# 9. SAVE DECISION TREE PNG
# ======================================================================

plt.figure(figsize=(30, 15))
plot_tree(dt, feature_names=features, filled=True, fontsize=10)
plt.savefig("output/decision_tree.png", dpi=300)
plt.close()


# ======================================================================
# 10. CONFUSION MATRIX PNG
# ======================================================================

cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("output/confusion_matrix.png", dpi=300)
plt.close()


# ======================================================================
# 11. FUZZY C-MEANS CLUSTERING
# ======================================================================

fuzzy_features = ["cp", "thalach", "oldpeak", "exang", "ca", "slope", "thal"]
X_fuzzy = df[fuzzy_features]
data = X_fuzzy.values.T

cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
    data, c=2, m=2.0, error=0.005, maxiter=1000
)

cluster_labels = np.argmax(u, axis=0)

df_compare = pd.DataFrame({
    "cluster": cluster_labels,
    "target": y.values
})

# Map fuzzy clusters → real disease classes
mapping = {}
for c in [0,1]:
    mapping[c] = df_compare[df_compare["cluster"] == c]["target"].mode()[0]

mapped = df_compare["cluster"].map(mapping)
alignment = (mapped.values == y.values).mean()

# Identify which cluster is disease
disease_cluster = 0 if mapping[0] == 1 else 1
disease_risk = u[disease_cluster]


# ======================================================================
# 12. FUZZY PLOTS (CORRECTED)
# ======================================================================

# ------ Plot 1: Membership Strength ------
plt.figure(figsize=(10,5))
plt.plot(u[0], label="Membership to Cluster 0 (u0)")
plt.plot(u[1], label="Membership to Cluster 1 (u1)")
plt.title("Fuzzy Membership Strength (u0 & u1)")
plt.xlabel("Sample Index")
plt.ylabel("Membership Value")
plt.legend()
plt.savefig("output/fuzzy_membership_strength.png", dpi=300)
plt.close()

# ------ Plot 2: cp vs oldpeak (hard cluster labels) ------
plt.figure(figsize=(7,5))
plt.scatter(X_fuzzy["cp"], X_fuzzy["oldpeak"], c=cluster_labels, cmap="coolwarm", alpha=0.7)
plt.xlabel("Chest Pain Type (cp)")
plt.ylabel("Oldpeak")
plt.title("Fuzzy Clusters (cp vs oldpeak)")
plt.savefig("output/fuzzy_feature_space.png", dpi=300)
plt.close()

# ------ Plot 3: TRUE Disease Risk Heatmap (CORRECT) ------
plt.figure(figsize=(7,5))
plt.scatter(
    X_fuzzy["thalach"],
    X_fuzzy["oldpeak"],
    c=disease_risk,
    cmap="viridis",
    alpha=0.8
)
plt.colorbar(label="Fuzzy Disease Risk (Corrected)")
plt.xlabel("Thalach (Heart Rate)")
plt.ylabel("Oldpeak")
plt.title("Corrected Fuzzy Disease Risk Heatmap")
plt.savefig("output/fuzzy_membership_intensity.png", dpi=300)
plt.close()


# ======================================================================
# 13. SYSTEM ARCHITECTURE PNG
# ======================================================================

plt.figure(figsize=(8,5))
plt.text(0.1, 0.85, "Input Data (Kaggle Heart Dataset)")
plt.text(0.1, 0.70, "↓ Scaling + Train/Test Split")
plt.text(0.1, 0.55, "↓ Decision Tree Classifier")
plt.text(0.1, 0.40, "↓ Confusion Matrix")
plt.text(0.1, 0.25, "↓ Fuzzy C-Means Validation")
plt.axis("off")
plt.savefig("output/system_architecture.png", dpi=300)
plt.close()


# ======================================================================
# 14. PDF REPORT
# ======================================================================

c = canvas.Canvas("output/full_report.pdf", pagesize=letter)
c.setFont("Helvetica", 12)

c.drawString(30, 750, "Heart Disease Prediction Report - Decision Tree")
c.drawString(30, 730, f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
c.drawString(30, 710, f"Fuzzy Alignment: {alignment:.4f}")
c.drawString(30, 690, f"Fuzzy Partition Coefficient (FPC): {fpc:.4f}")
c.drawString(30, 660, "Saved Files:")
c.drawString(40, 640, "decision_tree.png")
c.drawString(40, 620, "confusion_matrix.png")
c.drawString(40, 600, "fuzzy_membership_strength.png")
c.drawString(40, 580, "fuzzy_feature_space.png")
c.drawString(40, 560, "fuzzy_membership_intensity.png")
c.drawString(40, 540, "system_architecture.png")
c.drawString(40, 520, "tree_rules.txt")
c.drawString(40, 500, "decision_tree.pkl")

c.save()


# ======================================================================
# DONE
# ======================================================================

print("\nAll Decision Tree + Corrected Fuzzy outputs saved in 'output/' folder!")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Fuzzy Alignment:", alignment)
