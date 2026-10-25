import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("       NeuroKernel Train/Test Split")
print("=" * 60)

# Load prepared data
X = pd.read_csv("data/X_features.csv")
y = pd.read_csv("data/y_labels.csv")["label"]

print("\nOriginal data:")
print("X:", X.shape)
print("y:", y.shape)

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# Save the split datasets
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nSaved:")
print("data/X_train.csv")
print("data/X_test.csv")
print("data/y_train.csv")
print("data/y_test.csv")

print("\n" + "=" * 60)
print("Train/test split completed")
print("=" * 60)
