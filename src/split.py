import pandas as pd

def load_splits(path="data/creditcard.csv", train_frac=0.70, val_frac=0.15):
    df = pd.read_csv(path).sort_values("Time").reset_index(drop=True)
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train, val, test = df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]
    return train, val, test

def xy(d):
    return d.drop(columns="Class"), d["Class"]
