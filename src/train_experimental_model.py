from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/experimental_process_dataset.csv")
MODEL_PATH = Path("models/neurokernel_experimental_rf.pkl")

FEATURES = [
    "cpu_percent",
    "memory_mb",
    "memory_percent",
    "threads",
    "context_switches",
    "cpu_user_time",
    "cpu_system_time",
    "cpu_delta",
    "read_bytes_per_sec",
    "write_bytes_per_sec",
]

df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

print("Training experimental model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

print("\n===== EXPERIMENTAL MODEL RESULTS =====")
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("\nConfusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )
)
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, predictions), 4))
print("Anomaly average precision:", round(average_precision_score(y_test, probabilities), 4))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("\nExperimental model saved to:", MODEL_PATH)
print("Existing model was not overwritten.")
