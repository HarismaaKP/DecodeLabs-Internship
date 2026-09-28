"""
Project 2: Data Classification Using AI (DecodeLabs)
Goal: Build a classification model on the Iris dataset using KNN.
Pipeline: Input -> Process -> Output (IPO Framework)
"""

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score
)

# ---------------- INPUT ----------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
class_names = iris.target_names

print("Dataset shape:", X.shape)
print("Classes:", list(class_names))
print(X.head(), "\n")

# ---------------- PROCESS ----------------
# 1. Train-test split (80/20, shuffled, stratified to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Feature scaling (StandardScaler: mean=0, variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Find optimal K (elbow method on error rate, K >= 3 to avoid overfitting)
error_rates = []
k_range = range(1, 21)
for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    pred_k = knn_k.predict(X_test_scaled)
    error_rates.append(np.mean(pred_k != y_test))

candidates = [k for k in k_range if k >= 3]
best_k = candidates[int(np.argmin([error_rates[k - 1] for k in candidates]))]
print(f"Optimal K selected: {best_k}\n")

# 4. Train final KNN model with optimal K
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

# 5. Predict
predictions = model.predict(X_test_scaled)

# ---------------- OUTPUT ----------------
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
cm = confusion_matrix(y_test, predictions)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (macro): {f1:.4f}\n")
print("Confusion Matrix:\n", cm, "\n")
print("Classification Report:\n", classification_report(y_test, predictions, target_names=class_names))