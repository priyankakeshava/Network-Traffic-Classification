import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 📂 STEP 1: Load Dataset
# -----------------------------
df = pd.read_csv("cleaned_ddos_data.csv")
df.columns = df.columns.str.strip()

print("✅ Dataset loaded:", df.shape)

# -----------------------------
# 🎯 STEP 2: Separate Features and Label
# -----------------------------
X = df.drop(columns=['Label'])
y = df['Label']

# -----------------------------
# 🔢 STEP 3: Encode Labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\n📌 Label Mapping:")
for i, label in enumerate(le.classes_):
    print(f"{label} -> {i}")

# -----------------------------
# 🔀 STEP 4: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Size:", X_train.shape)
print("Testing Size:", X_test.shape)

# -----------------------------
# 🌳 STEP 5: Train Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model trained successfully")

# -----------------------------
# 📈 STEP 6: Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 📊 STEP 7: Evaluation Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Accuracy:", accuracy)

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📋 Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=le.classes_
    )
)

# -----------------------------
# 💾 STEP 8: Save Model
# -----------------------------
joblib.dump(model, "ddos_model.pkl")
print("\n✅ Model saved as ddos_model.pkl")

# =============================
# 📊 VISUALIZATION 1:
# Class Distribution
# =============================
plt.figure(figsize=(8, 5))
sns.countplot(x='Label', data=df)

plt.title("Class Distribution")
plt.xlabel("Traffic Type")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# =============================
# 📊 VISUALIZATION 2:
# Confusion Matrix Heatmap
# =============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# =============================
# 📊 VISUALIZATION 3:
# Feature Importance
# =============================
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(
    range(len(indices)),
    importances[indices]
)

plt.yticks(
    range(len(indices)),
    X.columns[indices]
)

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()
plt.show()

# =============================
# 📊 VISUALIZATION 4:
# Feature vs Label Boxplot
# =============================

# Change this feature name if needed
feature_name = "Packet Length Mean"

if feature_name in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x='Label',
        y=feature_name,
        data=df
    )

    plt.title(f"{feature_name} vs Label")
    plt.xlabel("Traffic Type")
    plt.ylabel(feature_name)

    plt.tight_layout()
    plt.show()

else:
    print(f"\n⚠ Column '{feature_name}' not found. Skipping boxplot.")

# =============================
# 📊 VISUALIZATION 5:
# Correlation Heatmap
# =============================
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()

print("\n✅ All visualizations generated successfully.")
