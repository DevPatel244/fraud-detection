from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ensemble import ensemble_proba
from src.features import to_features

ART = joblib.load(Path(__file__).resolve().parent.parent / "models" / "ensemble.joblib")
app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    transaction_id: str
    features: dict[str, float]  # V1..V28 and Amount (Time is ignored)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": ART["threshold"]}


@app.post("/predict")
def predict(t: Transaction):
    X = to_features(pd.DataFrame([t.features]))
    missing = [c for c in ART["features"] if c not in X.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing features: {missing}")
    p = float(ensemble_proba(ART["models"], X[ART["features"]])[0])
    return {
        "transaction_id": t.transaction_id,
        "fraud_probability": round(p, 4),
        "is_fraud": p >= ART["threshold"],
        "threshold": ART["threshold"],
    }
