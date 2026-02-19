import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
data = pd.read_csv("datasets/email.csv")

print("Columns in dataset:", data.columns)
print("Class distribution:\n", data["label"].value_counts())

# -----------------------------
# 2️⃣ Features & Labels
# -----------------------------
X = data["text_combined"]
y = data["label"]

# -----------------------------
# 3️⃣ Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4️⃣ Text Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5️⃣ Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -----------------------------
# 6️⃣ Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7️⃣ Save Model + Vectorizer
# -----------------------------
os.makedirs("models_storage", exist_ok=True)

joblib.dump(model, "models_storage/email_model.pkl")
joblib.dump(vectorizer, "models_storage/email_vectorizer.pkl")

print("\n✅ Email model saved successfully!")
