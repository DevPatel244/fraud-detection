def ensemble_proba(models, X):
    """Soft voting: average the fraud probability of each model."""
    return sum(m.predict_proba(X)[:, 1] for m in models) / len(models)
