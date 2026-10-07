import numpy as np
from src.ensemble import ensemble_proba

class Const:
    def __init__(self, p):
        self.p = p
    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1 - p, p])

def test_soft_voting_is_the_mean():
    out = ensemble_proba([Const(0.2), Const(0.4), Const(0.9)], np.zeros((3, 2)))
    assert np.allclose(out, 0.5)
