import numpy as np
from src.split import xy

def to_features(X):
    """Raw transaction columns -> model features. Used in training AND the API."""
    X = X.drop(columns=["Time", "Class"], errors="ignore").copy()
    X["Amount"] = np.log1p(X["Amount"])
    return X

def prepare(d):
    X, y = xy(d)
    return to_features(X), y
