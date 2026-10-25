
import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

print("=" * 60)
print("     NeuroKernel Anomaly-Type Analysis")
print("=" * 60)

# Load model
model = joblib.load("models/neurokernel_random_forest.pkl")

# Load original dataset
df = pd.read_csv("data/neurokernel_final.csv")

features = [
    "cpu_percent",
    "memory_mb",
    "memory_percent",
    "threads",
    "context_switches",
    "cpu_user_time",
    "cpu_system_time",
    "cpu_delta",
    "read_bytes_per_sec",
    "write_bytes_per_sec"
]

X = df[features]
y = df["label"]

# Recreate the same stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Get anomaly types for exactly the same test rows
test_types = df.loc[X_test.index, "anomaly_type"].reset_index(drop=True)

# Reset indices so rows align
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Predictions
predictions = model.predict(X_test)

results = pd.DataFrame({
    "actual_label": y_test,
    "predicted_label": predictions,
    "anomaly_type": test_types
})

print("\nOverall confusion matrix:")
print(confusion_matrix(
    results["actual_label"],
    results["predicted_label"],
    labels=[0, 1]
))

print("\nOverall classification report:")
print(classification_report(
    results["actual_label"],
    results["predicted_label"],
    labels=[0, 1],
    target_names=["Normal", "Anomaly"],
    zero_division=0
))

print("\nPerformance by anomaly type:")

for anomaly_type in results["anomaly_type"].unique():
    subset = results[results["anomaly_type"] == anomaly_type]

    print("\n" + "-" * 50)
    print("Type:", anomaly_type)
    print("Samples:", len(subset))

    print(classification_report(
        subset["actual_label"],
        subset["predicted_label"],
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

print("\n" + "=" * 60)
print("Analysis completed")
print("=" * 60)

