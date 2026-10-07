import pandas as pd
from src.split import load_splits

def test_splits_are_chronological(tmp_path):
    df = pd.DataFrame({"Time": range(100, 0, -1), "Amount": 1.0, "Class": 0})
    f = tmp_path / "d.csv"
    df.to_csv(f, index=False)
    tr, va, te = load_splits(f)
    assert tr["Time"].max() <= va["Time"].min()
    assert va["Time"].max() <= te["Time"].min()
    assert len(tr) + len(va) + len(te) == 100
