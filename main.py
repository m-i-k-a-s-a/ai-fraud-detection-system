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

# ==========================================
# TEMPLATE SETUP
# ==========================================

templates = Jinja2Templates(directory="templates")

# ==========================================
# DATABASE SETUP
# ==========================================

Base.metadata.create_all(bind=engine)

# ==========================================
# MODEL FOLDER
# ==========================================

MODEL_FOLDER = "models_storage"

# ==========================================
# LOAD MODELS
# ==========================================

transaction_model = joblib.load(os.path.join(MODEL_FOLDER, "transaction_model.pkl"))

email_model = joblib.load(os.path.join(MODEL_FOLDER, "email_model.pkl"))
email_vectorizer = joblib.load(os.path.join(MODEL_FOLDER, "email_vectorizer.pkl"))

phishing_model = joblib.load(os.path.join(MODEL_FOLDER, "phishing_model.pkl"))





# ==========================================
# ROUTES (HTML)
# ==========================================

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





# ==========================================
# REQUEST MODELS
# ==========================================

class Transaction(BaseModel):
    features: list[float]

class EmailRequest(BaseModel):
    text: str

class PhishingURLRequest(BaseModel):
    url: str

# ==========================================
# TRANSACTION FRAUD
# ==========================================

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

# ==========================================
# EMAIL PHISHING
# ==========================================

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

# ==========================================
# WEBSITE PHISHING
# ==========================================

@app.post("/predict_phishing")
def predict_phishing(request: PhishingURLRequest):
    try:
        features = extract_features(request.url)

        if len(features) != 30:
            return {"error": "Feature extraction failed"}

        data = np.array(features).reshape(1, -1)
        phishing_probability = float(phishing_model.predict_proba(data)[0][1])

        risk_score = round(phishing_probability * 100, 2)

        if phishing_probability > 0.5:
            result = "Phishing Website Detected 🚨"
        else:
            result = "Website Safe ✅"

        return {
            "result": result,
            "risk_score": risk_score
        }

    except Exception as e:
        return {"error": str(e)}

