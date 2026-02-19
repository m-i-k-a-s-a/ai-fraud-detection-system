from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from database import engine, SessionLocal, Base
from models import TransactionLog
import joblib
import numpy as np
import os
from utils.url_feature_extractor import extract_features

app = FastAPI()

# ------------------------------
# Setup Templates
# ------------------------------

templates = Jinja2Templates(directory="templates")

# ------------------------------
# Create Database Tables
# ------------------------------

Base.metadata.create_all(bind=engine)

# ------------------------------
# Model Folder
# ------------------------------

MODEL_FOLDER = "models_storage"

# ------------------------------
# Load Models
# ------------------------------

transaction_model = joblib.load(os.path.join(MODEL_FOLDER, "transaction_model.pkl"))
email_model = joblib.load(os.path.join(MODEL_FOLDER, "email_model.pkl"))
email_vectorizer = joblib.load(os.path.join(MODEL_FOLDER, "email_vectorizer.pkl"))
phishing_model = joblib.load(os.path.join(MODEL_FOLDER, "phishing_model.pkl"))

# ------------------------------
# Routes
# ------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/transaction", response_class=HTMLResponse)
def transaction_page(request: Request):
    return templates.TemplateResponse("transaction.html", {"request": request})


@app.get("/email", response_class=HTMLResponse)
def email_page(request: Request):
    return templates.TemplateResponse("email.html", {"request": request})


@app.get("/phishing", response_class=HTMLResponse)
def phishing_page(request: Request):
    return templates.TemplateResponse("phishing.html", {"request": request})

# ------------------------------
# Request Models
# ------------------------------

class Transaction(BaseModel):
    features: list[float]


class EmailRequest(BaseModel):
    text: str


class PhishingURLRequest(BaseModel):
    url: str

# ------------------------------
# Transaction Prediction
# ------------------------------

@app.post("/predict")
def predict_transaction(transaction: Transaction):
    try:
        data = np.array(transaction.features).reshape(1, -1)

        score = transaction_model.decision_function(data)[0]
        threshold = 0.14937210608876408

        if score < threshold:
            result = "Fraud Detected 🚨"
        else:
            result = "Transaction Safe ✅"

        db = SessionLocal()
        log = TransactionLog(
            result=result,
            risk_score=round(float(score), 4)
        )
        db.add(log)
        db.commit()
        db.close()

        return {
            "result": result,
            "risk_score": round(float(score), 4)
        }

    except Exception as e:
        return {"error": str(e)}

# ------------------------------
# Email Prediction
# ------------------------------

@app.post("/predict_email")
def predict_email(request: EmailRequest):
    try:
        text_vector = email_vectorizer.transform([request.text])
        prediction = email_model.predict(text_vector)[0]

        if prediction == 1:
            result = "Phishing Email Detected 🚨"
        else:
            result = "Email Safe ✅"

        return {"result": result}

    except Exception as e:
        return {"error": str(e)}

# ------------------------------
# Phishing Feature Names
# ------------------------------

PHISHING_FEATURE_NAMES = [
    "Having IP Address",
    "URL Length",
    "Shortening Service",
    "Having @ Symbol",
    "Double Slash Redirecting",
    "Prefix-Suffix (- in domain)",
    "Having Sub Domain",
    "SSL Final State",
    "Domain Registration Length",
    "Favicon",
    "Port",
    "HTTPS Token",
    "Request URL",
    "URL of Anchor",
    "Links in Tags",
    "SFH",
    "Submitting to Email",
    "Abnormal URL",
    "Redirect",
    "on_mouseover",
    "RightClick Disabled",
    "Popup Window",
    "Iframe",
    "Age of Domain",
    "DNS Record",
    "Web Traffic",
    "Page Rank",
    "Google Index",
    "Links Pointing to Page",
    "Statistical Report"
]

# ------------------------------
# Phishing Website Prediction
# ------------------------------

@app.post("/predict_phishing")
def predict_phishing(request: PhishingURLRequest):
    try:
        features = extract_features(request.url)

        if len(features) != 30:
            return {"error": f"Feature extraction failed. Expected 30 features, got {len(features)}"}

        data = np.array(features).reshape(1, -1)

        # Get phishing probability
        phishing_probability = float(phishing_model.predict_proba(data)[0][1])

        risk_score = round(phishing_probability * 100, 2)

        # Feature importance explanation
        importances = phishing_model.feature_importances_
        feature_importance_pairs = list(zip(importances, PHISHING_FEATURE_NAMES, features))
        feature_importance_pairs.sort(key=lambda x: x[0], reverse=True)

        explanations = []

        for importance, name, value in feature_importance_pairs:
            if value == -1:
                explanations.append(f"{name} strongly indicates phishing behavior")
            elif value == 0:
                explanations.append(f"{name} appears suspicious")

        explanations = explanations[:5]

        return {
            "risk_score": risk_score,
            "explanation": explanations
        }

    except Exception as e:
        return {"error": str(e)}
