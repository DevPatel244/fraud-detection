import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("\nClass counts:\n", df["Class"].value_counts())
print(f"\nFraud rate: {df['Class'].mean():.4%}")
print("\nAmount by class:\n", df.groupby("Class")["Amount"].describe())
print("\nTime span (hours):", df["Time"].max() / 3600)
print("Sorted by Time:", df["Time"].is_monotonic_increasing)
