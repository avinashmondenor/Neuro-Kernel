import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("        NeuroKernel Random Forest Training")
print("=" * 60)

# Load training data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv")["label"]
y_test = pd.read_csv("data/y_test.csv")["label"]

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

print("Training completed.")

# Predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Anomaly"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\nFeature Importance:")

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print(importance.to_string(index=False))

# Save model
joblib.dump(model, "models/neurokernel_random_forest.pkl")

print("\nModel saved as:")
print("models/neurokernel_random_forest.pkl")

print("\n" + "=" * 60)
print("NeuroKernel model training completed")
print("=" * 60)
