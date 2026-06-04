import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 📂 Load dataset
# -----------------------------
df = pd.read_csv("cleaned_ddos_data.csv")
df.columns = df.columns.str.strip()

print("✅ Dataset loaded:", df.shape)

# -----------------------------
# 🎯 Prepare data
# -----------------------------
X = df.drop(columns=['Label'])
y = df['Label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# 🔀 Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

# -----------------------------
# 🌳 Train model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained")

# -----------------------------
# 📊 Predictions
# -----------------------------
y_pred = model.predict(X_test)

# =============================
# 📊 1. CLASS DISTRIBUTION
# =============================
plt.figure()
sns.countplot(x='Label', data=df)
plt.title("Class Distribution")
plt.xlabel("Traffic Type")
plt.ylabel("Count")
plt.show()

# =============================
# 📊 2. CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =============================
# 📊 3. FEATURE IMPORTANCE
# =============================
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# =============================
# 📊 4. FEATURE VS LABEL (BOXPLOT)
# =============================
plt.figure()
sns.boxplot(x='Label', y='Packet Length Mean', data=df)
plt.title("Packet Length Mean vs Label")
plt.xlabel("Traffic Type")
plt.ylabel("Packet Length Mean")
plt.show()

# =============================
# 📊 5. CORRELATION HEATMAP (FIXED)
# =============================
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
