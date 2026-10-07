import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "models" / "ensemble.joblib").exists():
    pytest.skip("trained model not present", allow_module_level=True)

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def _load(name):
    return json.loads((ROOT / "tests" / name).read_text())

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_fraud_sample_is_flagged():
    assert client.post("/predict", json=_load("sample_fraud.json")).json()["is_fraud"] is True

def test_normal_sample_is_not_flagged():
    assert client.post("/predict", json=_load("sample_normal.json")).json()["is_fraud"] is False

def test_missing_features_rejected():
    r = client.post("/predict", json={"transaction_id": "x", "features": {"Amount": 1.0}})
    assert r.status_code == 422
