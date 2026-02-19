import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

print("Loading dataset...")

# Load dataset
data = pd.read_csv("fraud.csv")

print("Dataset loaded successfully!")

# Remove target column if exists
if "Class" in data.columns:
    X = data.drop("Class", axis=1)
else:
    X = data

print("Training model...")

# Train Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X)

print("Saving model...")

# Save model
joblib.dump(model, "transaction_model.pkl")

print("Model trained and saved!")
