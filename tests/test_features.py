import numpy as np
import pandas as pd
from src.features import to_features

def test_drops_time_and_class_and_logs_amount():
    X = pd.DataFrame({"Time": [1.0], "V1": [0.5], "Amount": [99.0], "Class": [0]})
    out = to_features(X)
    assert list(out.columns) == ["V1", "Amount"]
    assert np.isclose(out["Amount"].iloc[0], np.log1p(99.0))
