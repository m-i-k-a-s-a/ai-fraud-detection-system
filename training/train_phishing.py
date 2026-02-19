import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------
# Load Dataset
# ------------------------------

data = pd.read_csv("datasets/phishing.csv")

print("Original Dataset shape:", data.shape)

# ------------------------------
# Remove Unnecessary Columns
# ------------------------------

# Drop index column
if "index" in data.columns:
    data = data.drop("index", axis=1)

print("After dropping index:", data.shape)

# ------------------------------
# Prepare Data
# ------------------------------

# Convert labels: -1 (phishing) → 1
data["Result"] = data["Result"].apply(lambda x: 1 if x == -1 else 0)

X = data.drop("Result", axis=1)
y = data["Result"]

print("\nTotal Features Used:", X.shape[1])
print("Class distribution:\n", y.value_counts())

# ------------------------------
# Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Train Model
# ------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------
# Evaluate
# ------------------------------

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------
# Save Model
# ------------------------------

os.makedirs("models_storage", exist_ok=True)

joblib.dump(model, "models_storage/phishing_model.pkl")

print("\n✅ Phishing model saved successfully!")
